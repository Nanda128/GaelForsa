import React, { useEffect, useState } from 'react';

function ErrorMessage({ message, duration = 5000, onClose }) {
    const [visible, setVisible] = useState(true);

    useEffect(() => {
        if (!message) return;

        const timer = setTimeout(() => {
            setVisible(false);
            if (onClose) onClose();
        }, duration);

        return () => clearTimeout(timer);
    }, [message, duration, onClose]);

    if (!message || !visible) return null;

    return (
        <div className="error-message">
            {message}
        </div>
    );
}

export default ErrorMessage;

