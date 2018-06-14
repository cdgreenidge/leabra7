$ErrorActionPreference = "Stop"

If ($env:APPVEYOR_REPO_BRANCH -ne "master")  {
  echo "Not on master branch, canceling deployment to Anaconda Cloud."
  Exit 0
} ElseIf (Test-Path env:APPVEYOR_PULL_REQUEST_NUMBER) {
  echo "This is a pull request, canceling deployment to Anaconda Cloud."
  Exit 0
} else {
  echo "Deploying to Anaconda Cloud..."
}

echo "Finding package path..."
$packagePath = (conda build .\scripts\meta.yaml --output)

echo "Uploading package..."
anaconda --token $env:CONDA_TOKEN upload $packagePath