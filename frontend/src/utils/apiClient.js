const API_BASE_URL = '/api';

async function request(endpoint, options = {}) {
    const url = `${API_BASE_URL}${endpoint}`;
    const config = {
        headers: {
            'Content-Type': 'application/json', ...options.headers
        }, ...options
    };

    let response;
    try {
        // const startTime = performance.now();
        response = await fetch(url, config);
        // console.log(`${options.method || 'GET'} ${endpoint} took ${performance.now() - startTime}ms`);
    } catch (error) {
        if (error.name === 'TypeError' && error.message.includes('fetch')) {
            throw new Error('Network error: Unable to connect to server');
        }
        throw error;
    }

    if (!response.ok) {
        throw new Error(`Request failed: ${response.status} ${response.statusText}`);
    }

    const contentType = response.headers.get('content-type');
    // if (contentType?.includes('application/json')) {
    if (contentType && contentType.includes('application/json')) {
        return await response.json();
    }

    return await response.text();
}

export function get(endpoint, params = {}) {
    const queryString = new URLSearchParams(params).toString();
    const url = queryString ? `${endpoint}?${queryString}` : endpoint;
    return request(url, {method: 'GET'});
}

export function post(endpoint, data) {
    return request(endpoint, {
        method: 'POST', body: JSON.stringify(data)
    });
}

export function patch(endpoint, data) {
    return request(endpoint, {
        method: 'PATCH', body: JSON.stringify(data)
    });
}

export function put(endpoint, data) {
    return request(endpoint, {
        method: 'PUT', body: JSON.stringify(data)
    });
}

export function del(endpoint) {
    return request(endpoint, {method: 'DELETE'});
}

export async function upload(endpoint, formData) {
    const url = `${API_BASE_URL}${endpoint}`;

    let response;
    try {
        response = await fetch(url, {
            method: 'POST', body: formData
        });
    } catch (error) {
        if (error.name === 'TypeError' && error.message.includes('fetch')) {
            throw new Error('Network error: Unable to connect to server');
        }
        throw error;
    }

    if (!response.ok) {
        throw new Error(`Upload failed: ${response.status} ${response.statusText}`);
    }

    const contentType = response.headers.get('content-type');
    if (contentType && contentType.includes('application/json')) {
        return await response.json();
    }

    return await response.text();
}

