if (Test-Path env:APPVEYOR_PULL_REQUEST_NUMBER) {
   echo "Pull request detected. Aborting deployment..."
   exit 0
}

echo "Finding package path..."
$packagePath = (conda build scripts/meta.yaml --output)

echo "Uploading package..."
anaconda --token $env:CONDA_TOKEN upload $packagePath