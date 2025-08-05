echo "add dir!!!"
read dir
echo "add comment!!!"
read comment

git add ./$dir
git commit -m "$comment"
git push origin main