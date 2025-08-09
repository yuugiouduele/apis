echo "add dir!!!"
read dir
comment=`head /dev/urandom | tr -dc A-Za-z | head -c 8; echo`

git add ./$dir
git commit -m "$comment"
git push origin main