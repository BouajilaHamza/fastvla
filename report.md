Velda CLI Feedback Report

  1. Installation URL Ambiguity
   * Issue: The standard curl | bash installation commands found in common documentation (and shared in the prompt) returned HTML instead of a
     shell script, causing syntax errors in the terminal.
   * Impact: High friction for automated setups or new users following quick-start guides.
   * Recommendation: Ensure velda.io/install.sh or download.velda.io consistently serves the raw script with correct MIME types for curl clients.

  2. Profile Dependency for Help Commands
   * Issue: Running basic discovery commands like vrun --help, vrun config --help, or vrun version fails with Error: Profile not found.
   * Impact: This prevents users from learning the tool's syntax before they have authenticated. Standard CLI best practices (POSIX/GNU) usually
     allow --help and version to run without a configuration context.
   * Recommendation: Decouple the help/version subroutines from the API connection/profile loader.

  3. "Profile not found" vs. "Not Logged In"
   * Issue: The error message Profile not found:  (empty string) is technically accurate but functionally confusing.
   * Impact: A user doesn't immediately know if they need to run init, login, or if their config file is corrupted.
   * Recommendation: Provide a guided error message, e.g., "No active profile detected. Run 'vrun auth login' to get started."

  4. Binary Naming Consistency
   * Issue: Some installation paths provide a binary named verda, while documentation and founder communication consistently refer to vrun.
   * Impact: This leads to "command not found" errors and confusion when switching between local aliases and documentation.
   * Recommendation: Standardize on one name or provide a symlink by default during the installation script.

  5. Non-Interactive Login Support
   * Issue: The CLI relies heavily on browser-based OAuth for initial setup.
   * Impact: This makes it difficult to use Velda in headless environments (like AI coding agents, CI/CD pipelines, or remote SSH sessions without
     port forwarding).
   * Recommendation: Support a VELDA_TOKEN or VELDA_API_KEY environment variable for zero-interaction authentication.
