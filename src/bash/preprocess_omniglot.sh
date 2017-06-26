# from omniglot/python directory
for i in *.zip; do unzip $i; done      
mkdir all_images 
find -not -path "all_images/*" -name '*.png' | xargs mv -t all_images/
for i in all_images/*.png; do echo $i; convert -trim -resize 105x105 -gravity center -extent 105x105 -negate -resize 28x28 $i $i; done
mkdir testing-images
mkdir testing-images/0
mkdir training-images
mkdir training-images/0
mv all_images/*_{01..06}.png testing-images/0/
mv all_images/{1362..1623}_07.png testing-images/0/
echo number of files in testing is $(ls testing-images/0 | wc -l)
mv all_images/*.png training-images/0
rmdir all_images

python ~/eccentricity/src/python/convert-images-to-mnist-format.py

