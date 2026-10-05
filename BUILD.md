# TENRYU β ビルド手順

## 計算サーバ要件

- Linux x86_64
- NVIDIA GPU（sm_80 以上を推奨）
- CUDA Toolkit 12.6
- GCC 12 以上
- CMake 3.27 以上
- Ninja
- Python 3.10 以上（開発用ヘッダを含む）。pybind11 は導入済みのものを使い、無ければ CMake が configure 時に取得します（ネットワークが要ります）
- HDF5

## ビルドと動作確認

計算サーバ上で次のコマンドを実行します。

```bash
git clone <beta-repo> TENRYU && cd TENRYU
export PATH=/usr/local/cuda/bin:$PATH   # nvcc にパスを通す（CUDA Toolkit の既定の場所）
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=$(which python3)
ninja -C build tenryu
./build/tenryu run examples/verification/sod_planar.py   # 動作確認
```

## GUI 接続

サーバ上の `build/tenryu` の絶対パスを、
TENRYU Studio のサーバ設定に登録してください。
実行ディレクトリには、接続ユーザーが書き込める任意のパスを指定します。

## トラブルシュート

- configure が `No CMAKE_CUDA_COMPILER could be found` で止まる: nvcc にパスが通っていません。NVIDIA の手順で導入した CUDA Toolkit は nvcc を `/usr/local/cuda/bin` に置くので、`export PATH=/usr/local/cuda/bin:$PATH` を実行してから configure し直すか、CMake に `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc` を付けてください
- pybind11 の取得に失敗する（configure する計算機がネットワークに出られないなど）: pybind11 を先に導入してから configure し直してください。Ubuntu 24.04 では `apt install pybind11-dev`（2.11.1）、pip ならネットワークに出られる環境で `python3 -m pip install pybind11`（Ubuntu 24.04 などシステムの Python への導入を pip が拒む場合は、仮想環境を作るか `--break-system-packages` を付ける）。Catch2・spdlog・CLI11 も、無ければ同じく configure 時に取得します
- HDF5 が見つからない: Debian/Ubuntu では `libhdf5-dev` を導入
- コンパイルが `Killed` で止まる: メモリ不足でコンパイラが強制終了されています。Ninja は既定で CPU 数 + 2 個のコンパイルを同時に走らせ、使うメモリもその数とともに増えるため、CPU 数に比べてメモリの少ない計算機（コンテナのメモリ上限を含む）で起こります。`ninja -C build -j 8 tenryu` のように同時実行数を減らして実行し直してください
- GPU architecture: 既定では configure 時に `nvidia-smi` でローカル GPU を検出し、その compute capability のみをビルドします（GPU が見えないホストでは可搬既定 `70;80;89;90`）。明示指定するときは CMake に `-DCMAKE_CUDA_ARCHITECTURES=<num>` を追加
