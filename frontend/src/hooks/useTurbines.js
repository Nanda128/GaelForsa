import { useState, useEffect, useCallback } from 'react';
import { fetchTurbines } from '../utils/api';

export function useTurbines() {
    const [turbines, setTurbines] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const loadTurbines = useCallback(async () => {
        try {
            setLoading(true);
            setError(null);
            const data = await fetchTurbines();
            // console.log('Turbines loaded:', data.length);
            setTurbines(data);
        } catch (err) {
            setError(err.message);
            console.error('Failed to load turbines:', err);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        loadTurbines();
    }, [loadTurbines]);

    return {
        turbines,
        loading,
        error,
        refreshTurbines: loadTurbines
    };
}

