"""Behavior tests for local --model directories and streamed --corpus-parquet text.

These tests must not load model weights or contact the Hugging Face hub.
"""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import neuron_stats


def _write_parquet(path, values, row_group_size=None):
    table = pa.table({"text": values})
    if row_group_size is None:
        pq.write_table(table, path)
    else:
        pq.write_table(table, path, row_group_size=row_group_size)


class NeuronStatsCorpusTests(unittest.TestCase):
    def _model_dir(self, root, config='{"model_type": "qwen3_5"}', tokenizer=True):
        path = Path(root) / "model"
        path.mkdir()
        if config is not None:
            (path / "config.json").write_text(config, encoding="utf-8")
        if tokenizer:
            (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        return str(path)

    def _parse_error(self, argv):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit):
                neuron_stats.build_parser().parse_args(argv)
        return stderr.getvalue()

    def _stop_before_weights(self):
        calls = []

        def tok(*args, **kwargs):
            calls.append(("tok", args, kwargs))
            return object()

        def model(*args, **kwargs):
            calls.append(("model", args, kwargs))
            raise RuntimeError("stop before weights")

        return calls, tok, model

    def test_hub_model_id_is_not_forced_local_and_does_not_load_weights(self):
        # Break: treating the default hub id as a filesystem path, or passing local_files_only.
        calls, tok, model = self._stop_before_weights()
        argv = ["neuron_stats.py", "--model", "Qwen/Qwen3.5-4B"]
        with patch.object(sys, "argv", argv), \
                patch.object(neuron_stats.AutoTokenizer, "from_pretrained", tok), \
                patch.object(neuron_stats.AutoModelForImageTextToText, "from_pretrained", model):
            with self.assertRaisesRegex(RuntimeError, "stop before weights"):
                neuron_stats.main()
        self.assertEqual([item[0] for item in calls], ["tok", "model"])
        for _, args, kwargs in calls:
            self.assertEqual(args[0], "Qwen/Qwen3.5-4B")
            self.assertIs(kwargs["local_files_only"], False)

    def test_local_model_directory_uses_local_files_only_without_loading_weights(self):
        # Break: from_pretrained still hits the hub for an on-disk model directory.
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = self._model_dir(tmp)
            calls, tok, model = self._stop_before_weights()
            argv = ["neuron_stats.py", "--model", model_dir]
            with patch.object(sys, "argv", argv), \
                    patch.object(neuron_stats.AutoTokenizer, "from_pretrained", tok), \
                    patch.object(neuron_stats.AutoModelForImageTextToText, "from_pretrained", model):
                with self.assertRaisesRegex(RuntimeError, "stop before weights"):
                    neuron_stats.main()
            self.assertEqual([item[0] for item in calls], ["tok", "model"])
            for _, args, kwargs in calls:
                self.assertEqual(os.path.abspath(args[0]), os.path.abspath(model_dir))
                self.assertIs(kwargs["local_files_only"], True)

    def test_missing_local_model_directory_errors_before_weights(self):
        # Break: a typo'd absolute model path falls through to a hub lookup.
        missing = "/tmp/neuron-stats-missing-model-dir"
        self.assertFalse(os.path.exists(missing))
        with patch.object(
            neuron_stats.AutoTokenizer, "from_pretrained", side_effect=AssertionError("loaded weights")
        ):
            message = self._parse_error(["--model", missing])
        self.assertIn(missing, message)
        self.assertIn("not found", message.lower())

    def test_model_directory_missing_config_errors(self):
        # Break: a directory without a model config is handed to from_pretrained.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty"
            path.mkdir()
            (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            message = self._parse_error(["--model", str(path)])
        self.assertIn("config.json", message)

    def test_model_directory_missing_tokenizer_errors(self):
        # Break: config-only folders are accepted as loadable model directories.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model"
            path.mkdir()
            (path / "config.json").write_text('{"model_type": "qwen3_5"}', encoding="utf-8")
            message = self._parse_error(["--model", str(path)])
        self.assertIn("tokenizer", message.lower())

    def test_model_path_that_is_a_file_errors(self):
        # Break: a file path is treated as a model directory.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text("{}", encoding="utf-8")
            message = self._parse_error(["--model", str(path)])
        self.assertIn(str(path), message)
        self.assertIn("director", message.lower())

    def test_invalid_model_config_json_errors(self):
        # Break: unreadable config.json reaches weight loading.
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = self._model_dir(tmp, config="{")
            message = self._parse_error(["--model", model_dir])
        self.assertIn("config.json", message)

    def test_cli_omits_corpus_parquet_by_default(self):
        # Break: the FineWeb + UltraChat path is no longer the default.
        args = neuron_stats.build_parser().parse_args([])
        self.assertEqual(args.model, "Qwen/Qwen3.5-4B")
        self.assertIsNone(args.corpus_parquet)

    def test_parquet_text_streams_in_file_order_across_row_groups(self):
        # Break: rows are sorted, shuffled, or collapsed across row-group boundaries.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus.parquet"
            _write_parquet(path, ["zeta", "alpha", "mu"], row_group_size=1)
            self.assertEqual(
                list(neuron_stats.iter_parquet_text(str(path))),
                ["zeta", "alpha", "mu"],
            )

    def test_parquet_iteration_is_batched_and_does_not_read_the_full_table(self):
        # Break: the reader calls read_table / ParquetFile.read, or ignores batch size.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus.parquet"
            _write_parquet(path, ["r0", "r1", "r2", "r3", "r4"], row_group_size=1)
            real_iter = pq.ParquetFile.iter_batches

            def spy(self, *args, **kwargs):
                spy.calls.append(kwargs)
                return real_iter(self, *args, **kwargs)

            spy.calls = []

            def forbid_table(*args, **kwargs):
                raise AssertionError("full parquet table was loaded")

            with patch.object(pq.ParquetFile, "iter_batches", spy), \
                    patch.object(pq, "read_table", forbid_table), \
                    patch.object(pq.ParquetFile, "read", forbid_table), \
                    patch.object(pq.ParquetFile, "read_row_groups", forbid_table):
                texts = list(neuron_stats.iter_parquet_text(str(path), batch_size=2))
            self.assertEqual(texts, ["r0", "r1", "r2", "r3", "r4"])
            self.assertGreaterEqual(len(spy.calls), 1)
            for kwargs in spy.calls:
                self.assertEqual(kwargs.get("columns"), ["text"])
                self.assertEqual(kwargs.get("batch_size"), 2)

    def test_parquet_batch_sizes_follow_the_requested_batch(self):
        # Break: iter_batches is called once with a batch that swallows the whole file.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus.parquet"
            _write_parquet(path, ["r0", "r1", "r2", "r3", "r4"], row_group_size=1)
            seen = []
            real_iter = pq.ParquetFile.iter_batches

            def spy(self, *args, **kwargs):
                def track():
                    for batch in real_iter(self, *args, **kwargs):
                        seen.append(batch.num_rows)
                        yield batch

                return track()

            with patch.object(pq.ParquetFile, "iter_batches", spy):
                texts = list(neuron_stats.iter_parquet_text(str(path), batch_size=2))
            self.assertEqual(texts, ["r0", "r1", "r2", "r3", "r4"])
            self.assertEqual(seen, [2, 2, 1])

    def test_parquet_skips_nulls_and_keeps_file_order_for_any_seed(self):
        # Break: nulls are yielded, or seed shuffles the parquet stream.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus.parquet"
            _write_parquet(path, ["zeta", None, "alpha"], row_group_size=1)

            class Tok:
                def apply_chat_template(self, *args, **kwargs):
                    raise AssertionError("chat template should not run for parquet text")

            with patch.object(neuron_stats, "load_dataset", side_effect=AssertionError("hub corpus")):
                first = list(neuron_stats.text_stream(Tok(), 0, corpus_parquet=str(path)))
                second = list(neuron_stats.text_stream(Tok(), 99, corpus_parquet=str(path)))
            self.assertEqual(first, ["zeta", "alpha"])
            self.assertEqual(second, ["zeta", "alpha"])

    def test_windows_pack_parquet_texts_in_file_order(self):
        # Break: windows() ignores --corpus-parquet and still opens the hub corpora.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus.parquet"
            _write_parquet(path, ["ab", "cd"])

            class Tok:
                eos_token_id = 9

                def __call__(self, text, add_special_tokens=False):
                    return {"input_ids": [ord(ch) for ch in text]}

            with patch.object(neuron_stats, "load_dataset", side_effect=AssertionError("hub corpus")):
                windows = list(neuron_stats.windows(Tok(), seq_len=4, batch=1, seed=7, corpus_parquet=str(path)))
            self.assertEqual(len(windows), 1)
            self.assertEqual(windows[0].tolist(), [[ord("a"), ord("b"), 9, ord("c")]])

    def test_default_stream_remains_fineweb_plus_ultrachat(self):
        # Break: omitting --corpus-parquet no longer interleaves those two corpora.
        calls = []

        class FakeSplit:
            def __init__(self, rows):
                self.rows = rows

            def shuffle(self, seed):
                self.seed = seed
                return self.rows

        def fake_load(name, *args, **kwargs):
            calls.append((name, args, kwargs))
            if name == "HuggingFaceFW/fineweb-edu":
                return FakeSplit([{"text": "web-doc"}])
            if name == "HuggingFaceH4/ultrachat_200k":
                return FakeSplit([{"messages": [{"role": "user", "content": "hi"}]}])
            raise AssertionError(name)

        class Tok:
            def apply_chat_template(self, messages, tokenize=False, enable_thinking=False):
                self.messages = messages
                self.tokenize = tokenize
                self.enable_thinking = enable_thinking
                return "CHAT"

        tok = Tok()
        with patch.object(neuron_stats, "load_dataset", fake_load):
            texts = list(neuron_stats.text_stream(tok, 3))
        self.assertEqual(texts, ["web-doc", "CHAT"])
        self.assertEqual(calls[0][0], "HuggingFaceFW/fineweb-edu")
        self.assertEqual(calls[0][1], ("sample-10BT",))
        self.assertEqual(calls[0][2]["split"], "train")
        self.assertIs(calls[0][2]["streaming"], True)
        self.assertEqual(calls[1][0], "HuggingFaceH4/ultrachat_200k")
        self.assertEqual(calls[1][1], ())
        self.assertEqual(calls[1][2]["split"], "train_sft")
        self.assertIs(calls[1][2]["streaming"], True)
        self.assertEqual(tok.messages, [{"role": "user", "content": "hi"}])
        self.assertIs(tok.tokenize, False)
        self.assertIs(tok.enable_thinking, False)

    def test_parquet_missing_text_column_errors(self):
        # Break: a parquet file with no text column is streamed as empty or crashes later.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus.parquet"
            pq.write_table(pa.table({"content": ["hello"]}), path)
            message = self._parse_error(["--corpus-parquet", str(path)])
        self.assertIn("text", message)
        self.assertIn("content", message)

    def test_parquet_text_column_wrong_type_errors(self):
        # Break: a non-string text column is accepted.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus.parquet"
            pq.write_table(pa.table({"text": [1, 2]}), path)
            message = self._parse_error(["--corpus-parquet", str(path)])
        self.assertIn("text", message)
        self.assertTrue("int" in message.lower() or "string" in message.lower())

    def test_parquet_missing_file_errors(self):
        # Break: a missing parquet path is opened lazily inside the training loop.
        missing = "/tmp/neuron-stats-missing-corpus.parquet"
        self.assertFalse(os.path.exists(missing))
        message = self._parse_error(["--corpus-parquet", missing])
        self.assertIn(missing, message)
        self.assertIn("not found", message.lower())

    def test_parquet_directory_errors(self):
        # Break: a directory is accepted as a parquet file.
        with tempfile.TemporaryDirectory() as tmp:
            message = self._parse_error(["--corpus-parquet", tmp])
        self.assertIn(tmp, message)

    def test_parquet_unreadable_file_errors(self):
        # Break: a non-parquet file produces a pyarrow traceback instead of an argument error.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "notes.parquet"
            path.write_text("not parquet", encoding="utf-8")
            message = self._parse_error(["--corpus-parquet", str(path)])
        self.assertIn(str(path), message)

    def test_dictionary_encoded_text_column_streams_values(self):
        # Break: dictionary-encoded string columns are rejected or decoded out of order.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus.parquet"
            values = pa.array(["zeta", "alpha"]).dictionary_encode()
            pq.write_table(pa.table({"text": values}), path, row_group_size=1)
            self.assertEqual(list(neuron_stats.iter_parquet_text(str(path))), ["zeta", "alpha"])

    def test_empty_text_column_streams_nothing(self):
        # Break: an empty text column raises instead of yielding an empty stream.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus.parquet"
            pq.write_table(pa.table({"text": pa.array([], type=pa.string())}), path)
            self.assertEqual(list(neuron_stats.iter_parquet_text(str(path))), [])


if __name__ == "__main__":
    unittest.main()
