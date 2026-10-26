# Release policy

- Keep GitHub's default branch on the newest version that has passed acceptance. Update it whenever a newer version is ready.
- Mark each newly published stable release as Latest. Keep unpublished versions labelled as previews.
- Publish each new version with an increasing version number, a new Git tag, and a separate GitHub Release.
- Publish the same version to GitHub, PyPI, and Zenodo as one release. On Zenodo, add a new version to the existing record with its own DOI.
- Update README.md and README.zh-CN.md for the release, including the version, relevant usage changes, installation links, and Zenodo archive link.
- Verify all three public releases, their files, and the README links before reporting the release complete. If a destination is blocked, report it as incomplete and resume when access is available.
- Keep all previously published tags, release records, and assets unchanged. Never move an existing tag, delete an old release, or replace its files.
- Ship corrections in the next version. Do not reuse a published version number or overwrite an existing package on any registry.
- Check existing remote tags and releases before choosing a version. Use the version number alone as the release title.
