// BikeGuard Updates Page JavaScript
class UpdatesManager {
  constructor() {
    this.GITHUB_USER = 'medhansh5';
    this.GITHUB_REPO = 'BikeGuard';
    this.timeline = document.getElementById('timeline');
    this.fallbackData = [
      {
        date: '2026-05-01',
        version: 'v0.3.0-alpha',
        badge: 'upcoming',
        title: 'Pillion Detection Model Training',
        description: 'Advanced multi-rider detection system with independent compliance tracking and seat position classification.',
        features: [
          'Multi-rider compliance tracking',
          'Secondary seat position classifier',
          'Expanded QA suite coverage',
          'Multi-camera input support'
        ]
      },
      {
        date: '2026-04-15',
        version: 'v0.2.0-alpha',
        badge: 'prerelease',
        title: 'Core Architecture Scaffolded',
        description: 'Complete system architecture implementation with all core modules active and passing QA tests.',
        features: [
          'Core architecture fully scaffolded and pushed to GitHub',
          'Real-time helmet detection compliance passing QA tests',
          'Sikh turban exemption logic implemented',
          'Pillion rider detection scaffolded',
          'DirectML GPU-accelerated inference pipeline initialized',
          'Royal Enfield Classic 350 "The Baron" calibration profile active',
          'Telemetry logging system active'
        ]
      }
    ];
    
    this.init();
  }
  
  init() {
    this.showShimmers();
    this.loadUpdates();
  }
  
  showShimmers() {
    // Show shimmer placeholders while loading
    for (let i = 0; i < 3; i++) {
      const shimmer = document.createElement('div');
      shimmer.className = 'shimmer';
      this.timeline.appendChild(shimmer);
    }
  }
  
  async loadUpdates() {
    try {
      // Try to fetch from GitHub API
      const response = await fetch(`https://api.github.com/repos/${this.GITHUB_USER}/${this.GITHUB_REPO}/releases`);
      
      if (response.ok) {
        const releases = await response.json();
        this.renderTimeline(releases);
      } else {
        throw new Error('GitHub API not available');
      }
    } catch (error) {
      // Fallback to static data
      setTimeout(() => {
        this.renderTimeline(this.fallbackData);
      }, 1000);
    }
  }
  
  renderTimeline(data) {
    // Clear shimmers
    this.timeline.innerHTML = '';
    
    data.forEach((item, index) => {
      const timelineItem = this.createTimelineItem(item, index);
      this.timeline.appendChild(timelineItem);
    });
    
    // Setup intersection observer for staggered animation
    this.setupIntersectionObserver();
  }
  
  createTimelineItem(item, index) {
    const div = document.createElement('div');
    div.className = 'timeline-item';
    
    // Handle GitHub API releases vs fallback data
    const isGitHubRelease = item.published_at !== undefined;
    const formattedDate = this.formatDate(isGitHubRelease ? item.published_at : item.date);
    const badgeClass = this.getBadgeClass(isGitHubRelease ? (item.prerelease ? 'prerelease' : 'stable') : (item.badge || 'prerelease'));
    const badgeText = this.getBadgeText(isGitHubRelease ? (item.prerelease ? 'PRERELEASE' : 'STABLE') : (item.badge || 'prerelease'));
    
    // Add special styling for upcoming items
    if (item.badge === 'upcoming') {
      div.classList.add('timeline-item-upcoming');
    }
    
    const title = isGitHubRelease ? item.name : `${item.version} — ${item.title}`;
    const description = isGitHubRelease ? (item.body ? '' : 'No description provided.') : item.description;
    const features = isGitHubRelease ? this.parseBody(item.body) : (item.features ? item.features.map(feature => `<li>${feature}</li>`).join('') : '');
    
    div.innerHTML = `
      <div class="timeline-date">${formattedDate}</div>
      <div class="timeline-content">
        <div class="timeline-badge ${badgeClass}">${badgeText}</div>
        <div class="timeline-title">${title}</div>
        ${description ? `<div class="timeline-description">${description}</div>` : ''}
        ${features ? `<ul class="timeline-features">${features}</ul>` : ''}
      </div>
    `;
    
    return div;
  }
  
  formatDate(isoString) {
    if (!isoString) return 'No date';
    const d = new Date(isoString);
    if (isNaN(d.getTime())) return 'No date';
    const yyyy = d.getUTCFullYear();
    const mm = String(d.getUTCMonth() + 1).padStart(2, '0');
    const dd = String(d.getUTCDate()).padStart(2, '0');
    return `${yyyy}-${mm}-${dd}`;
  }
  
  parseBody(body) {
    if (!body || body.trim() === '') return '<li>No details provided.</li>';
    return body
      .split(/\r?\n/)
      .map(line => line.replace(/^[-*]\s*/, '').trim())
      .filter(line => line.length > 0)
      .map(line => `<li>${line}</li>`)
      .join('');
  }
  
  getBadgeClass(type) {
    switch (type) {
      case 'stable': return 'badge-stable';
      case 'upcoming': return 'badge-upcoming';
      default: return 'badge-prerelease';
    }
  }
  
  getBadgeText(type) {
    switch (type) {
      case 'stable': return 'STABLE';
      case 'upcoming': return 'UPCOMING';
      default: return 'PRERELEASE';
    }
  }
  
  setupIntersectionObserver() {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
        }
      });
    }, {
      threshold: 0.1,
      rootMargin: '0px 0px -50px 0px'
    });
    
    document.querySelectorAll('.timeline-item').forEach(item => {
      observer.observe(item);
    });
  }
}

// Initialize updates manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  if (document.getElementById('timeline')) {
    new UpdatesManager();
  }
});
