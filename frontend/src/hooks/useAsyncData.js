import { useState, useEffect } from 'react';

export function useAsyncData(fetchFunction, dependencies = []) {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        async function loadData() {
            try {
                setLoading(true);
                setError(null);
                const result = await fetchFunction();
                setData(result);
            } catch (err) {
                setError(err.message);
                console.error('Failed to load data:', err);
            } finally {
                setLoading(false);
            }
        }
        
        loadData();
    }, dependencies);

    return { data, loading, error };
}

