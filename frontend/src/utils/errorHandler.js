export function handleApiError(error, defaultMessage = 'An error occurred') {
    if (error instanceof Error) {
        return error.message;
    }
    if (typeof error === 'string') {
        return error;
    }
    return defaultMessage;
}

export function isNetworkError(error) {
    return error?.message?.includes('Failed to fetch') || 
           error?.message?.includes('NetworkError') ||
           error?.code === 'NETWORK_ERROR';
}

export function getErrorMessage(error, context = '') {
    const baseMessage = handleApiError(error);
    return context ? `${context}: ${baseMessage}` : baseMessage;
}

