"""Fixed analytical markers inside the shared FOV observed in the initial pass."""
import pathlib, json, hashlib
from PIL import Image, ImageDraw
out=pathlib.Path(__file__).parent/'camera-capture-v3'
out.mkdir(exist_ok=True)
image=Image.new('RGB',(1920,1440),(220,220,220)); draw=ImageDraw.Draw(image)
for x in range(0,1920,120): draw.line((x,0,x,1440),fill=(170,170,170),width=2)
for y in range(0,1440,120): draw.line((0,y,1920,y),fill=(170,170,170),width=2)
marks=[]
for name,x,y,rgb in [('red',450,600,(230,30,40)),('magenta',650,600,(215,35,195)),('cyan',650,840,(20,180,200)),('blue',450,840,(35,65,230))]:
    draw.ellipse((x-20,y-20,x+20,y+20),fill=rgb)
    marks.append({'id':name,'center':[x,y],'radius':20,'rgb':list(rgb)})
draw.text((300,400),'TOP / ANALYTICAL FEED V3',fill=(0,0,0),font_size=30)
image.save(out/'landmark-feed.png')
record={'schema_version':1,'scope':'isolated_synthetic_camera_geometry_only',
        'generator_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        'image':'landmark-feed.png','sha256':hashlib.sha256((out/'landmark-feed.png').read_bytes()).hexdigest(),
        'size':list(image.size),'markers':marks,'tolerance_photo_pixels':4,'tolerance_preview_pixels':3,
        'placement':'Moved analytical markers into shared FOV observed in first pass; no recognition model, predicted word, or geometry transform fitting. Fixed before new capture. Original feed/evidence retained separately.'}
(out/'feed-manifest.json').write_text(json.dumps(record,indent=2)+'\n')
print(record['sha256'])
