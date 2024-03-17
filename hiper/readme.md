#　Hiper

[Highly Personalized Text Embedding for Image Manipulation by Stable Diffusion](https://arxiv.org/abs/2303.08767)の再現実装

必要なライブラリ

- transformers
- diffusers
- Pillow
- tqdm
- torch
- torchvision

使い方
```bash
python main.py --guidance 7.5 --optim adam 
```

`main`関数を編集してsource promptとtarget promptを変更可能。`--infer`をつけずに実行するとpersonalizeして生成まで行う。つけると`optim_emb.pt`を読んで推論する。