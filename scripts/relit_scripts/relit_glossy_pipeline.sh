output_dir=$1

sh relit_scripts/relit_glossy.sh $output_dir
sh relit_scripts/eval_relit_glossy.sh $output_dir
sh relit_scripts/avg_relit_glossy.sh $output_dir