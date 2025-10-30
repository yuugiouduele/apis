docker ps --format "{{.ID}} {{.Image}}" | while read container_id image_name; do
  docker commit $container_id $image_name
  docker push kyukyoku/$image_name
  docker stop $container_id
done

docker system prune -a --volumes -f 
docker system df
docker ps 
docker images