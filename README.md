# Mpeg Audio Stem Extraction and Mixing

このプロジェクトは、指定されたフォルダ（`C:\Users\cammy\Documents\MpegSoundExtraction`）内の MPEG や MP4 などの動画・音声ファイルから音声を抽出し、Demucs を用いて各楽器（ボーカル、ドラム、ベース、ギター、ピアノ、その他）に高精度で分離します。その後、各パートの相対的な音量を調整し、新しい音声ファイルとしてリミックスします。

## 準備

プロジェクトの依存パッケージのインストールと実行は `uv` を使用して行います。
また、Python、FFmpeg、および NVIDIA GPU (CUDA対応) がセットアップされていることが前提です。

## 使い方

1. **対象ファイルの配置**
   処理したい動画ファイル（`.mpeg`, `.mp4`, `.m4a` 等）を `C:\Users\cammy\Documents\MpegSoundExtraction` フォルダに配置します。

2. **スクリプトの実行**
   ターミナル（PowerShell等）を開き、プロジェクトフォルダに移動して以下のコマンドを実行します。

   ```powershell
   cd C:\Users\cammy\Documents\MpegSoundExtraction
   uv run process_audio.py
   ```

   ※ `uv run` を使用することで、初回実行時に `pyproject.toml` に定義された依存関係（PyTorch等）が自動的に仮想環境へインストールされ、実行されます。

## 音量バランスの調整

`process_audio.py` をテキストエディタで開き、スクリプト上部の `RELATIVE_VOLUMES` の数値を変更することで、各楽器の音量を相対的に調整できます。

```python
RELATIVE_VOLUMES = {
    'vocals': 1.0,
    'drums': 1.0,
    'bass': 1.0,
    'guitar': 1.5,  # ギターを大きめに設定
    'piano': 1.0,
    'other': 1.0
}
```

## 注意事項

- **初回実行時のダウンロード**: 初回実行時は Demucs の高精度モデル（`htdemucs_6s`）が自動的にダウンロードされるため、インターネット接続が必要です。
- **処理時間**: 分離処理には GPU のリソースを多く使用します（設定により CUDA を利用して高速化されます）。
- **出力ファイル**: 最終的なミックスは `元のファイル名_final_mix.wav` として保存され、各楽器の個別音源（Stem）は `separated/htdemucs_6s/元のファイル名/` フォルダに残ります。
