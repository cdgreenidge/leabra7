echo "Finding package path..."
$packagePath = (conda build .\scripts\meta.yaml --output)

echo "Uploading package..."
anaconda --token $env:CONDA_TOKEN upload $packagePath