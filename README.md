# 3D Rasterization Engine

Developed a Rasterization Engine with a Phong model. The data set is of an aneurysm and the colors correspond to pressure on the artery.

![](https://github.com/aturanb/3D-Rasterization-Engine/out_aneurysm.gif)

## Run

Compile the code, then execute:

```bash
gcc rasterization.c -lm -o rasterization
./rasterization
```
This generates 1000 frames. And these frames can be converted to a short mp4 file by running:
```bash
ffmpeg -f image2 -i rasterizer_frame%004d.pnm rasterizer.mp4