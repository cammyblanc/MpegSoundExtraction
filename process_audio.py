import os
import glob
import subprocess
import soundfile as sf
import numpy as np
import torch

# ============== 設定 (Configuration) ==============
# 各楽器の相対ボリューム（1.0を基準とする）
# ギター(guitar)の数値を変更することでバランスを自由に調整可能
RELATIVE_VOLUMES = {
    'vocals': 1.0,      # ボーカル
    'drums': 0.8,    # ドラム
    'bass':1.0,        # ベース
    'guitar': 2.0,      # ギター（例: 1.5倍に強調）
    'piano': 1.0,       # ピアノ
    'other': 1.0        # その他
}

# 処理対象のファイル拡張子
SUPPORTED_EXTENSIONS = ['*.mpeg', '*.mpg', '*.mp4', '*.ts', '*.mkv', '*.avi', '*.mp3', '*.wav', '*.m4a']
# ===================================================

def check_gpu():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPUが見つかりました: {gpu_name} を使用して高速処理を行います。")
        return "cuda"
    else:
        print("⚠️ GPUが見つかりません。CPUで処理を行いますが、非常に時間がかかります。")
        return "cpu"

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        print("❌ エラー: 'ffmpeg' コマンドが見つかりません。Demucsの実行にはffmpegが必要です。")
        print("   以下の手順でFFmpegをインストールしてください。")
        print("   Windows: ターミナルで `winget install ffmpeg` を実行し、再起動してください。")
        return False

def process_file(filepath, device):
    filename = os.path.basename(filepath)
    name, _ = os.path.splitext(filename)
    output_dir = "separated"
    
    print(f"\n🎧 処理開始: {filename}")
    
    # 1. Demucsで音源分離 (htdemucs_6sモデルを指定して6パートに分離)
    print("🥁 ボーカル・楽器の分離を行っています... (Demucs実行)")
    print("   ※ 初回実行時はDemucsのAIモデル(htdemucs_6s)のダウンロードが行われます。")
    cmd = [
        "demucs",
        "-n", "htdemucs_6s",
        "--device", device,
        "-o", output_dir,
        filepath
    ]
    
    # subprocessでDemucsを実行
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"❌ {filename} のDemucs分離処理でエラーが発生しました。")
        return
        
    stem_dir = os.path.join(output_dir, "htdemucs_6s", name)
    if not os.path.exists(stem_dir):
         print(f"❌ 分離されたフォルダ {stem_dir} が見つかりません。")
         return
         
    # 2. 分離されたStemを読み込み、相対ボリュームを考慮してミックス
    print("🎛️ 音量バランスを調整してリミックスしています...")
    
    mixed_audio = None
    sample_rate = None
    
    for stem_name, rel_vol in RELATIVE_VOLUMES.items():
        stem_path = os.path.join(stem_dir, f"{stem_name}.wav")
        if not os.path.exists(stem_path):
            print(f"⚠️ 警告: {stem_path} が見つかりません。スキップします。")
            continue
            
        audio_data, sr = sf.read(stem_path)
        
        if sample_rate is None:
            sample_rate = sr
            mixed_audio = np.zeros_like(audio_data)
        elif sr != sample_rate:
             print(f"⚠️ {stem_name} のサンプリングレートが異なります。")
             continue
             
        # 相対ボリュームを乗算してミックス
        print(f"  - {stem_name} 波形を合成中 (Volume: x{rel_vol})")
        mixed_audio += audio_data * rel_vol
        
    if mixed_audio is not None:
        # 3. クリッピング（音割れ）防止の正規化 (Normalize)
        # ミックスにより閾値(1.0)を超えた場合のみ、全体をスケールダウンする
        max_amplitude = np.max(np.abs(mixed_audio))
        if max_amplitude > 1.0:
            print(f"🔊 音割れ防止のため、全体のボリュームを正規化しています (Peak: {max_amplitude:.2f})")
            mixed_audio /= max_amplitude
            
        # 4. 最終結果の保存
        final_mix_path = f"{name}_final_mix.wav"
        sf.write(final_mix_path, mixed_audio, sample_rate)
        print(f"✨ 完了: 最終ミックスを {final_mix_path} に保存しました！")
        
        # 5. 元の動画と新しい音声を結合して新しいファイルを作成 (元の拡張子が動画系の場合)
        _, ext = os.path.splitext(filename)
        if ext.lower() not in ['.mp3', '.wav', '.m4a']:
            output_video_path = f"{name}_SoundMixed{ext}"
            print(f"🎬 元の動画と新しい音声を結合しています... -> {output_video_path}")
            
            # 元動画の映像のみ(0:v)と、作成したwavの音声(1:a)を合わせる
            merge_cmd = [
                "ffmpeg", "-y",
                "-i", filepath,
                "-i", final_mix_path,
                "-c:v", "copy",     # 映像は再エンコードせずにそのままコピー(高速)
                "-map", "0:v:0?",   # 入力0(元動画)の最初の映像トラックを取得（存在しない場合は無視）
                "-map", "1:a:0",    # 入力1(最終ミックス)の最初の音声トラックを取得
                output_video_path
            ]
            
            result_merge = subprocess.run(merge_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if result_merge.returncode == 0:
                print(f"🎉 成功: 音声差し替え済みの動画を {output_video_path} に保存しました！")
            else:
                print(f"❌ 警告: {output_video_path} の生成（FFmpeg結合）に失敗しました。")

def main():
    print("=== Mpeg Audio Stem Separation and Mixing ===")
    
    if not check_ffmpeg():
        return
        
    device = check_gpu()
    
    # 対象ファイルの検索
    target_files = []
    for ext in SUPPORTED_EXTENSIONS:
        # 大文字小文字を区別せず検索するため、os.listdirでマッチングさせる方が確実ですが
        # Windowsのglobは大文字小文字を区別しません
        target_files.extend(glob.glob(ext))
        
    if not target_files:
        print("📁 カレントディレクトリに処理対象の動画・音声ファイルが見つかりません。")
        print("   Mpegファイル等をこのスクリプトと同じフォルダに置いてください。")
        return
        
    for file in target_files:
        process_file(file, device)

if __name__ == "__main__":
    main()
