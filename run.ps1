$dockerComposeFile = "docker-compose.yml"

# Build the Docker images
Write-Host "Building Docker images..."
docker-compose -f $dockerComposeFile build

# Start the services using Docker Compose
Write-Host "Starting services..."
docker-compose -f $dockerComposeFile up

# Display the status of the running containers
Write-Host "Displaying the status of running containers..."
docker ps