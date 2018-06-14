$ErrorActionPreference = "Stop"

If ($env:APPVEYOR_REPO_BRANCH -eq "master")  {
    echo "Deploying to Anaconda Cloud..."
} else {
    echo "Not on master branch, canceling deployment to Anaconda Cloud."
    Exit 0
}

echo "Finding package path..."
$packagePath = (conda build ./scripts/meta.yaml --output)

echo "Uploading package..."
anaconda --token $env:CONDA_TOKEN upload $packagePath