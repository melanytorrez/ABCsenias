function highlightActiveLink() {
    const currentPath = window.location.pathname; 
    
    const navLinks = document.querySelectorAll('.nav-link');

    navLinks.forEach(link => {
        const linkPath = link.getAttribute('href'); 
        if (currentPath.includes(linkPath) && linkPath !== "") {
            link.classList.add('highlighted');
        } 
        else if (currentPath === '/' && link.getAttribute('data-page') === 'home') {
        }
    });
}