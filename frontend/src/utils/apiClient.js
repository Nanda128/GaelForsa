const API_BASE_URL = '/api/v1';

async function request(endpoint, options = {}) {
    const url = `${API_BASE_URL}${endpoint}`;
    const config = {
        headers: {
            'Content-Type': 'application/json',
            ...options.headers
        },
        ...options
    };

    try {
        // const startTime = performance.now();
        const response = await fetch(url, config);
        // console.log(`${options.method || 'GET'} ${endpoint} took ${performance.now() - startTime}ms`);
        
        if (!response.ok) {
            throw new Error(`Request failed: ${response.status} ${response.statusText}`);
        }
        
        const contentType = response.headers.get('content-type');
        // if (contentType?.includes('application/json')) {
        if (contentType && contentType.includes('application/json')) {
            return await response.json();
        }
        
        return await response.text();
    } catch (error) {
        if (error.name === 'TypeError' && error.message.includes('fetch')) {
            throw new Error('Network error: Unable to connect to server');
        }
        throw error;
    }
}

export function get(endpoint, params = {}) {
    const queryString = new URLSearchParams(params).toString();
    const url = queryString ? `${endpoint}?${queryString}` : endpoint;
    return request(url, { method: 'GET' });
}

export function post(endpoint, data) {
    return request(endpoint, {
        method: 'POST',
        body: JSON.stringify(data)
    });
}

export function patch(endpoint, data) {
    return request(endpoint, {
        method: 'PATCH',
        body: JSON.stringify(data)
    });
}

export function put(endpoint, data) {
    return request(endpoint, {
        method: 'PUT',
        body: JSON.stringify(data)
    });
}

export function del(endpoint) {
    return request(endpoint, { method: 'DELETE' });
}

