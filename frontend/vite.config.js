import { defineConfig } from 'vite';

export default defineConfig({
    // Development server config
    server: {
        port: 3000,
        open: true,
        proxy: {
            '/api': {
                target: 'http://localhost:8000',
                changeOrigin: true
            }
        }
    },

    // Build config
    build: {
        outDir: 'dist',
        sourcemap: true,
        assetsDir: 'assets'
    },

    preview: {
        port: 4173
    },

    // Test config (Vitest)
    test: {
        environment: 'jsdom',
        globals: true,
        include: ['src/**/*.{test,spec}.{js,ts}', 'tests/**/*.{test,spec}.{js,ts}']
    }
});

