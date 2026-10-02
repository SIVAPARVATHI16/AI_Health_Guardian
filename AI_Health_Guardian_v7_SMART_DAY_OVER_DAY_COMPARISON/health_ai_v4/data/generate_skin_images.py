import os,random,numpy as np
from PIL import Image,ImageDraw,ImageFilter
random.seed(7); np.random.seed(7)
OUT_DIR="../data/skin_images"; CLASSES=["healthy_skin","inflamed_rash","healing_rash"]; IMG_SIZE=128; N=150
TONES=[(224,172,132),(210,160,122),(235,185,145)]
def bg(size,tone):
    img=Image.new("RGB",(size,size),tone); arr=np.array(img).astype(np.int16)
    arr=np.clip(arr+np.random.randint(-4,4,arr.shape),0,255).astype(np.uint8)
    return Image.fromarray(arr).filter(ImageFilter.GaussianBlur(0.6))
def blotch(img,color,n,mn,mx,al=(80,160)):
    ov=Image.new("RGBA",img.size,(0,0,0,0)); d=ImageDraw.Draw(ov); w,h=img.size
    for _ in range(n):
        r=random.randint(mn,mx); cx,cy=random.randint(r,w-r),random.randint(r,h-r)
        d.ellipse([cx-r,cy-r,cx+r,cy+r],fill=color+(random.randint(*al),))
    return Image.alpha_composite(img.convert("RGBA"),ov.filter(ImageFilter.GaussianBlur(3))).convert("RGB")
def gen_healthy(s): return bg(s,random.choice(TONES))
def gen_inflamed(s):
    img=bg(s,random.choice(TONES))
    img=blotch(img,(215,20,20),random.randint(5,8),12,24,(160,220))
    return blotch(img,(235,70,50),random.randint(3,5),8,16,(140,200))
def gen_healing(s):
    img=bg(s,random.choice(TONES))
    img=blotch(img,(185,100,85),random.randint(5,7),11,19,(160,210))
    return blotch(img,(145,95,65),random.randint(2,4),4,9,(120,170))
GF={"healthy_skin":gen_healthy,"inflamed_rash":gen_inflamed,"healing_rash":gen_healing}
for cls in CLASSES:
    os.makedirs(f"{OUT_DIR}/{cls}",exist_ok=True)
    for i in range(N): GF[cls](IMG_SIZE).save(f"{OUT_DIR}/{cls}/{cls}_{i:03d}.png")
    print(f"  {cls}: {N} images")
print("Done.")
