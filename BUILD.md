# TENRYU β ビルド手順

## 計算サーバ要件

- Linux x86_64
- NVIDIA GPU（sm_80 以上を推奨）
- CUDA Toolkit 12.6
- GCC 12 以上
- CMake 3.27 以上
- Ninja
- Python 3.10 以上、および pip で導入した pybind11
- HDF5

## ビルドと動作確認

計算サーバ上で次のコマンドを実行します。

```bash
git clone <beta-repo> TENRYU && cd TENRYU
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=$(which python3)
ninja -C build tenryu
./build/tenryu run examples/verification/sod_planar.py   # 動作確認
```

## GUI 接続

サーバ上の `build/tenryu` の絶対パスを、
TENRYU Studio のサーバ設定に登録してください。
実行ディレクトリには、接続ユーザーが書き込める任意のパスを指定します。

## トラブルシュート

- pybind11 が見つからない: `python3 -m pip install pybind11`
- HDF5 が見つからない: Debian/Ubuntu では `libhdf5-dev` を導入
- GPU architecture: 既定では configure 時に `nvidia-smi` でローカル GPU を検出し、その compute capability のみをビルドします（GPU が見えないホストでは可搬既定 `70;80;89;90`）。明示指定するときは CMake に `-DCMAKE_CUDA_ARCHITECTURES=<num>` を追加
