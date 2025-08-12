docker ps --format "{{.ID}} {{.Image}}" | while read container_id image_name; do
  docker commit $container_id $image_name
  docker stop $container_id
done
