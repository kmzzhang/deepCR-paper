echo concatenating data segments to data.tar
cat testdata-chunk-* > data.tar
echo unzipping data.tar
tar -xvf data.tar
echo deleting data.tar
rm data.tar
