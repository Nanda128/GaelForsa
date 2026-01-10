import React from 'react';

function Button({ 
    children, 
    onClick, 
    variant = 'secondary', 
    type = 'button',
    disabled = false,
    className = '',
    ...props 
}) {
    const baseClass = 'btn';
    const variantClass = `btn-${variant}`;
    // const classes = [baseClass, variantClass, className].filter(Boolean).join(' ');
    
    return (
        <button
            className={`${baseClass} ${variantClass} ${className}`}
            onClick={onClick}
            type={type}
            disabled={disabled}
            {...props}
        >
            {children}
        </button>
    );
}

export default Button;

