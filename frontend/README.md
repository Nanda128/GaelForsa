# GaelFórsa Frontend

Frontend for the GaelFórsa Wind Turbine Monitoring platform, built with Vite and Leaflet.

## Development Info: Delete when done.

### Prerequisites

- Node.js 18+ and npm

### Setup

```bash
cd frontend
npm install
```

### Development Server

**Important:** The Django backend must be running for the API to work.

1. **Start Django backend** (in one terminal):
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   python manage.py runserver
   ```

2. **Start Vite frontend** (in another terminal):
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

This starts the Vite dev server at http://localhost:3000 with:
- Hot Module Replacement (HMR)
- API proxy to Django backend at http://localhost:8000

### Production Build

```bash
npm run build
```

Outputs optimized assets to `dist/` folder.

### Preview Production Build

```bash
npm run preview
```

## Adding New Features

### New Map Features

1. Create a new module in `src/map/`
2. Export from `src/map/index.js`
3. Import in `src/main.js` or other modules as needed

### New Components

1. Create JavaScript module in appropriate folder
2. Add styles in `src/styles/components/`
3. Import styles in `src/styles/main.css`

## API Configuration

The API base URL is automatically configured based on environment:
- Development: `http://localhost:8000/api/v1`
- Production: `/api/v1` (same origin)

This is handled in `src/utils/api.js`.

