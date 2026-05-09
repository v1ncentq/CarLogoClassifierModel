const scrollTopBtn = document.getElementById("scrollTopBtn");
const navbarBurger = document.querySelector(".navbar-burger");
const navbarMenu = document.querySelector(".navbar-menu");
const navLinks = document.querySelectorAll(".navbar-end .navbar-item[href^='#']");
const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const updateScrollTopButton = () => {
  if (!scrollTopBtn) {
    return;
  }

  scrollTopBtn.classList.toggle("show", window.scrollY > 500);
};

window.addEventListener("scroll", updateScrollTopButton, { passive: true });
updateScrollTopButton();

if (scrollTopBtn) {
  scrollTopBtn.addEventListener("click", () => {
    window.scrollTo({
      top: 0,
      behavior: prefersReducedMotion ? "auto" : "smooth",
    });
  });
}

if (navbarBurger && navbarMenu) {
  navbarBurger.addEventListener("click", () => {
    const isExpanded = navbarBurger.classList.toggle("is-active");

    navbarMenu.classList.toggle("is-active", isExpanded);
    navbarBurger.setAttribute("aria-expanded", String(isExpanded));
  });
}

navLinks.forEach((link) => {
  link.addEventListener("click", () => {
    navLinks.forEach((item) => item.classList.remove("is-active"));
    link.classList.add("is-active");

    if (navbarBurger && navbarMenu) {
      navbarBurger.classList.remove("is-active");
      navbarMenu.classList.remove("is-active");
      navbarBurger.setAttribute("aria-expanded", "false");
    }
  });
});

const revealSelectors = [
  ".section-head",
  ".project-copy",
  ".info-grid",
  ".info-card",
  ".pipeline-step",
  ".brand-strip",
  ".model-card",
  ".visual-result",
  ".result-description",
  ".demo-card",
  ".demo-actions",
  ".callout-box",
  ".brand-list li",
];

const revealTargets = document.querySelectorAll(revealSelectors.join(","));

if (!prefersReducedMotion && "IntersectionObserver" in window) {
  revealTargets.forEach((element, index) => {
    const delay = Math.min((index % 4) * 80, 240);

    element.classList.add("reveal");
    element.style.setProperty("--reveal-delay", `${delay}ms`);

    if (element.classList.contains("visual-result")) {
      element.dataset.reveal = "left";
    }

    if (element.classList.contains("result-description")) {
      element.dataset.reveal = "right";
    }
  });

  const revealObserver = new IntersectionObserver(
    (entries, observer) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) {
          return;
        }

        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      });
    },
    {
      rootMargin: "0px 0px -12% 0px",
      threshold: 0.16,
    }
  );

  revealTargets.forEach((element) => revealObserver.observe(element));
} else {
  revealTargets.forEach((element) => element.classList.add("is-visible"));
}

const sectionLinks = Array.from(navLinks)
  .map((link) => ({
    link,
    section: document.querySelector(link.hash),
  }))
  .filter(({ section }) => Boolean(section));

if ("IntersectionObserver" in window && sectionLinks.length > 0) {
  const linksBySectionId = new Map(
    sectionLinks.map(({ link, section }) => [section.id, link])
  );

  const sectionObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) {
          return;
        }

        navLinks.forEach((link) => link.classList.remove("is-active"));
        linksBySectionId.get(entry.target.id)?.classList.add("is-active");
      });
    },
    {
      rootMargin: "-45% 0px -48% 0px",
      threshold: 0,
    }
  );

  sectionLinks.forEach(({ section }) => sectionObserver.observe(section));
}
