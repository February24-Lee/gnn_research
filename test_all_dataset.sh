CONFIGS="gat_arxiv.yaml gat_citeseer.yaml gat_core_ml.yaml gat_flicker.yaml gat_products.yaml gat_pumbed.yaml"
for i in $CONFIGS
do
    python test.py -c $i
done