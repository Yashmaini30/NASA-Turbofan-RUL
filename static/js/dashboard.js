// Dashboard JavaScript for NASA Turbofan RUL Analysis

// Global variables
let charts = {};
let currentTheme = 'light';

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
    updateCurrentTime();
    setInterval(updateCurrentTime, 1000);
});

// Dashboard initialization
function initializeDashboard() {
    // Add smooth animations to elements
    animateElements();
    
    // Initialize tooltips
    initializeTooltips();
    
    // Initialize image modals
    setupImageModals();
    
    // Setup responsive handlers
    setupResponsiveHandlers();
    
    // Initialize search functionality
    setupSearch();
    
    console.log('🚀 NASA Turbofan RUL Dashboard initialized successfully!');
}

// Animate elements on page load
function animateElements() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry, index) => {
            if (entry.isIntersecting) {
                setTimeout(() => {
                    entry.target.classList.add('fade-in');
                }, index * 100);
            }
        });
    }, observerOptions);
    
    // Observe all cards and visualizations
    document.querySelectorAll('.stat-card, .analysis-card, .viz-card').forEach(el => {
        observer.observe(el);
    });
}

// Initialize Bootstrap tooltips
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Setup image modal functionality
function setupImageModals() {
    // Enhanced image modal with zoom and pan
    const imageModal = document.getElementById('imageModal');
    if (imageModal) {
        imageModal.addEventListener('show.bs.modal', function (event) {
            const modalImage = document.getElementById('modalImage');
            modalImage.style.transform = 'scale(1)';
            modalImage.style.cursor = 'zoom-in';
            
            modalImage.addEventListener('click', function() {
                if (this.style.transform === 'scale(1)' || this.style.transform === '') {
                    this.style.transform = 'scale(1.5)';
                    this.style.cursor = 'zoom-out';
                } else {
                    this.style.transform = 'scale(1)';
                    this.style.cursor = 'zoom-in';
                }
            });
        });
    }
}

// Global image modal function
function showImageModal(img) {
    const modal = document.getElementById('imageModal');
    const modalImage = document.getElementById('modalImage');
    const modalTitle = document.getElementById('imageModalTitle');
    
    if (modal && modalImage && modalTitle) {
        modalImage.src = img.src;
        modalImage.alt = img.alt;
        modalTitle.textContent = formatImageTitle(img.alt);
        
        const bsModal = new bootstrap.Modal(modal);
        bsModal.show();
    }
}

// Format image title for display
function formatImageTitle(filename) {
    return filename
        .replace(/^[^_]*_/, '') // Remove dataset prefix
        .replace(/_/g, ' ')      // Replace underscores with spaces
        .replace('.png', '')     // Remove file extension
        .split(' ')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

// Update current time in navbar
function updateCurrentTime() {
    const timeElement = document.getElementById('current-time');
    if (timeElement) {
        const now = new Date();
        timeElement.textContent = now.toLocaleTimeString();
    }
}

// Setup responsive handlers
function setupResponsiveHandlers() {
    let resizeTimeout;
    window.addEventListener('resize', function() {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(function() {
            // Redraw charts if they exist
            Object.values(charts).forEach(chart => {
                if (chart && typeof chart.resize === 'function') {
                    chart.resize();
                }
            });
        }, 250);
    });
}

// Setup search functionality
function setupSearch() {
    const searchInput = document.getElementById('searchInput');
    if (searchInput) {
        searchInput.addEventListener('input', debounce(performSearch, 300));
    }
}

// Debounce function for search
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Perform search functionality
function performSearch(event) {
    const query = event.target.value.toLowerCase();
    const cards = document.querySelectorAll('.stat-card, .analysis-card, .viz-card');
    
    cards.forEach(card => {
        const text = card.textContent.toLowerCase();
        if (text.includes(query)) {
            card.style.display = '';
            card.classList.add('search-highlight');
        } else {
            card.style.display = query ? 'none' : '';
            card.classList.remove('search-highlight');
        }
    });
}

// Loading state management
function showLoading(element) {
    if (element) {
        element.classList.add('loading');
        const spinner = document.createElement('div');
        spinner.className = 'spinner-border spinner-border-sm me-2';
        spinner.setAttribute('role', 'status');
        element.insertBefore(spinner, element.firstChild);
    }
}

function hideLoading(element) {
    if (element) {
        element.classList.remove('loading');
        const spinner = element.querySelector('.spinner-border');
        if (spinner) {
            spinner.remove();
        }
    }
}

// API Helper functions
async function fetchDatasetStats(datasetName) {
    try {
        showLoading(document.querySelector(`[data-dataset="${datasetName}"]`));
        
        const response = await fetch(`/api/dataset/${datasetName}/stats`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error fetching dataset stats:', error);
        showNotification('Error loading dataset statistics', 'error');
        return null;
    } finally {
        hideLoading(document.querySelector(`[data-dataset="${datasetName}"]`));
    }
}

// Notification system
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    notification.style.cssText = 'top: 100px; right: 20px; z-index: 9999; min-width: 300px;';
    
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 5000);
}

// Chart creation helpers (if Chart.js is available)
function createChart(canvasId, config) {
    const canvas = document.getElementById(canvasId);
    if (canvas && typeof Chart !== 'undefined') {
        if (charts[canvasId]) {
            charts[canvasId].destroy();
        }
        charts[canvasId] = new Chart(canvas, config);
        return charts[canvasId];
    }
    return null;
}

// Theme management
function toggleTheme() {
    currentTheme = currentTheme === 'light' ? 'dark' : 'light';
    document.body.setAttribute('data-theme', currentTheme);
    
    // Update charts if they exist
    Object.values(charts).forEach(chart => {
        if (chart && chart.options) {
            chart.options.plugins.legend.labels.color = currentTheme === 'dark' ? '#ffffff' : '#333333';
            chart.update();
        }
    });
    
    localStorage.setItem('dashboard-theme', currentTheme);
}

// Load saved theme
function loadSavedTheme() {
    const savedTheme = localStorage.getItem('dashboard-theme');
    if (savedTheme) {
        currentTheme = savedTheme;
        document.body.setAttribute('data-theme', currentTheme);
    }
}

// Export functionality
function exportVisualization(imageElement) {
    const link = document.createElement('a');
    link.download = formatImageTitle(imageElement.alt) + '.png';
    link.href = imageElement.src;
    link.click();
}

// Print functionality
function printDashboard() {
    window.print();
}

// Fullscreen functionality
function toggleFullscreen(element) {
    if (!document.fullscreenElement) {
        element.requestFullscreen().catch(err => {
            console.error('Error attempting to enable fullscreen:', err);
        });
    } else {
        document.exitFullscreen();
    }
}

// Keyboard shortcuts
document.addEventListener('keydown', function(event) {
    // Ctrl+F for search
    if (event.ctrlKey && event.key === 'f') {
        event.preventDefault();
        const searchInput = document.getElementById('searchInput');
        if (searchInput) {
            searchInput.focus();
        }
    }
    
    // Ctrl+P for print
    if (event.ctrlKey && event.key === 'p') {
        event.preventDefault();
        printDashboard();
    }
    
    // Escape to close modals
    if (event.key === 'Escape') {
        const activeModal = document.querySelector('.modal.show');
        if (activeModal) {
            const modalInstance = bootstrap.Modal.getInstance(activeModal);
            if (modalInstance) {
                modalInstance.hide();
            }
        }
    }
});

// Performance monitoring
function logPerformance() {
    if (performance.timing) {
        const loadTime = performance.timing.loadEventEnd - performance.timing.navigationStart;
        console.log(`📊 Dashboard loaded in ${loadTime}ms`);
    }
}

// Initialize performance logging
window.addEventListener('load', logPerformance);

// Error handling
window.addEventListener('error', function(event) {
    console.error('Dashboard error:', event.error);
    showNotification('An error occurred. Please refresh the page.', 'danger');
});

// Service worker registration (for future PWA features)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        // Future: Register service worker for offline capabilities
        console.log('🔧 Service worker support detected');
    });
}

// Export functions for global use
window.dashboardFunctions = {
    showImageModal,
    fetchDatasetStats,
    showNotification,
    toggleTheme,
    exportVisualization,
    printDashboard,
    toggleFullscreen
};
