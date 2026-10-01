"""MkDocs hooks that build the site from the repository root.

MkDocs rejects `docs_dir: .`, so the root is set here, after the config is
checked, and `exclude_docs` in mkdocs.yml picks the published files. Pages keep
their repository paths, so relative links work as they do on GitHub. Inline
links to files that are not published, or to directories without a README, go
to GitHub.
"""

import logging
import posixpath
import re
from pathlib import Path

log = logging.getLogger("mkdocs.hooks")
ROOT = Path(__file__).resolve().parent.parent
IMAGES = (".gif", ".jpeg", ".jpg", ".png", ".svg", ".webp")
# The target of an inline link that is a relative path: ](path#anchor)
LINK = re.compile(r"\]\((?![a-z][a-z0-9+.-]*:|#|/)([^)\s#]+)(#[^)\s]*)?\)")


def on_config(config):
    config.docs_dir = str(ROOT)
    return config


def on_serve(server, config, builder):
    # Watch the published sources only, not target/ or virtual environments.
    server.unwatch(config.docs_dir)
    for name in ("README.md", "CONTRIBUTING.md", "docs", "recipe", "src"):
        server.watch(str(ROOT / name))
    return server


def on_page_markdown(markdown, page, config, files):
    base = posixpath.dirname(page.file.src_uri)
    github = config.repo_url.rstrip("/")

    def published(path):
        file = files.get_file_from_path(path)
        return file is not None and file.inclusion.is_included()

    def fix(m):
        target, anchor = m[1], m[2] or ""
        path = posixpath.normpath(posixpath.join(base, target))
        if path.split("/")[0] == ".." or not (ROOT / path).exists() or published(path):
            return m[0]  # a page, or a missing file for MkDocs to report
        if (ROOT / path).is_dir():
            if published(posixpath.join(path, "README.md")):
                return f"]({posixpath.join(target, 'README.md')}{anchor})"
            return f"]({github}/tree/main/{path}{anchor})"
        if path.lower().endswith(IMAGES):
            # A GitHub page is not an image; the image has to be published.
            log.warning("%s: %s is not published, see exclude_docs", page.file.src_uri, target)
            return m[0]
        return f"]({github}/blob/main/{path}{anchor})"

    return LINK.sub(fix, markdown)
