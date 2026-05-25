$env:PATH += ";C:\Program Files\PostgreSQL\18\bin"
$env:PORT = "8080"
$env:DATABASE_URL = "postgresql://postgres:Johnpaulbtiu@localhost:5432/fakenews"
pnpm --filter @workspace/api-server run dev:local
