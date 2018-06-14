$ErrorActionPreference = "Stop"

$branch = git rev-parse --abbrev-ref HEAD
If ($branch -eq "master")  {
    echo "Deploying to Anaconda Cloud..."
} else {
    echo "Not on master branch, canceling deployment to Anaconda Cloud."
    Exit 0
}

echo "Finding package path..."
$packagePath = conda build . --output

echo "Uploading package..."
anaconda --token $CONDA_TOKEN upload $packagePath