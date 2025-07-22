output_dir=$1
scene_list="luyu angel bell cat horse potion tbell teapot"
for i in $scene_list; do
        python eval/metrics_relit_glossy.py \
        --img_paths ${output_dir}/${i} \
        --gt_paths data/glossy_relight/relight_gt/${i} 
done
