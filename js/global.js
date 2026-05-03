// BikeGuard Global JavaScript
document.addEventListener('DOMContentLoaded', function () {
  const sidebar = document.getElementById('sidebar');
  const navToggle = document.getElementById('nav-toggle');
  const body = document.body;
  
  // State management
  let isSidebarOpen = localStorage.getItem('bikeguard-sidebar-open') !== 'false';
  
  // Responsive: default to closed on mobile
  const isMobile = window.innerWidth <= 768;
  if (isMobile) {
    isSidebarOpen = false;
  }

  function setSidebar(open) {
    isSidebarOpen = open;
    
    if (open) {
      sidebar.classList.remove('collapsed');
      body.classList.remove('sidebar-collapsed');
      navToggle.innerHTML = '<span class="nav-toggle-icon">☰</span>';
    } else {
      sidebar.classList.add('collapsed');
      body.classList.add('sidebar-collapsed');
      navToggle.innerHTML = '<span class="nav-toggle-icon">☰</span>';
    }
    
    // Save state to localStorage
    localStorage.setItem('bikeguard-sidebar-open', open.toString());
  }

  // Toggle sidebar when nav toggle is clicked
  navToggle.addEventListener('click', function () {
    setSidebar(!isSidebarOpen);
  });
  
  // Handle window resize for responsive behavior
  window.addEventListener('resize', function() {
    const isNowMobile = window.innerWidth <= 768;
    if (isNowMobile && isSidebarOpen) {
      setSidebar(false);
    } else if (!isNowMobile && !isSidebarOpen) {
      setSidebar(true);
    }
  });

  // Initialize sidebar state
  setSidebar(isSidebarOpen);
  
  // Close sidebar when clicking outside on mobile
  document.addEventListener('click', function(e) {
    if (window.innerWidth <= 768 && isSidebarOpen) {
      if (!sidebar.contains(e.target) && !navToggle.contains(e.target)) {
        setSidebar(false);
      }
    }
  });
});
