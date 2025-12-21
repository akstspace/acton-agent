---
applyTo: "README.md,docs/**/*.md,**/*.md"
---

# Documentation Update Guidelines for Pull Requests

You are responsible for maintaining accurate, comprehensive documentation when code changes are made. Your task is to update existing documentation files to reflect changes introduced in pull requests.

## Your Mission

When a PR introduces changes to the codebase:
1. **Analyze the changes** - Understand what was added, modified, or removed
2. **Identify documentation impact** - Determine which docs need updates
3. **Update relevant documentation** - Modify existing docs to stay in sync with code
4. **Maintain consistency** - Keep the same style, tone, and structure as existing docs
5. **Add examples** - Include code examples for new features or changed behavior

## Critical: Documentation Philosophy

**Documentation should always reflect the CURRENT state of the code, not its history.**

✅ **DO:**
- Modify existing content in-place to reflect current functionality
- Update examples to show current API usage
- Replace outdated information with current information
- Treat the documentation as the current truth, not a historical record

❌ **DON'T:**
- Add markers like "New in version X.X" or "Added in PR #123"
- Use phrases like "we added", "newly available", "now supports"
- Create separate sections for "new features" vs "existing features"
- Mark changes with "Update:" or "Changed:" annotations
- Maintain version-specific documentation sections

**Example:**

❌ **Wrong:**
```markdown
## Features
- Basic authentication (existing)
- Token refresh (new in v2.0)
- OAuth support (added recently)
```

✅ **Correct:**
```markdown
## Features
- Basic authentication
- Token refresh
- OAuth support
```

---

## Step 1: Analyze the PR Changes

Carefully review:

**Code Changes:**
- New features, classes, functions, or methods added
- Modified function signatures, parameters, or return types
- Deprecated or removed functionality
- Changed behavior or logic
- New dependencies or requirements
- Configuration changes
- Breaking changes

**Testing Changes:**
- New test cases that reveal intended usage
- Modified tests that show changed behavior

**Existing Documentation:**
- Read current README.md
- Read all files in docs/
- Understand current documentation structure and style

---

## Step 2: Identify What Needs Updating

Determine which documentation sections require changes:

### README.md Updates Needed?

- **Installation:** New dependencies? Changed requirements?
- **Quick Start:** Does the minimal example still work?
- **Key Features:** New features to highlight? Removed features to delete?
- **Version/Badges:** Version bump needed?

### docs/ Updates Needed?

- **Getting Started:** Installation or setup changes?
- **Core Concepts:** New concepts introduced? Explanations outdated?
- **API Reference:** New or modified APIs? Parameter changes? New exceptions?
- **Examples:** Do existing examples still work? New examples needed?
- **Advanced Topics:** New patterns or best practices?

> **Note:** Since this is an experimental project, breaking changes are expected and normal. Focus on keeping documentation accurate to the current API rather than maintaining migration paths.

---

## Step 3: Update Documentation

### Updating Existing Content

**When modifying existing documentation:**

✅ **DO:**
- Match the existing writing style and tone
- Keep the same structure and formatting
- Preserve working examples (update them if needed)
- Cross-reference related changes in other sections
- Update code examples to reflect new syntax/patterns
- Verify examples still compile/run correctly
- **Modify content in-place** - don't mark things as "new" or "updated"
- **Write in present tense** - describe what the code does NOW

❌ **DON'T:**
- Change the overall documentation structure without good reason
- Remove examples unless they're completely obsolete
- Change terminology inconsistently
- Add incomplete or untested examples
- Break internal links between docs
- Change the tone or writing style drastically
- **Add temporal markers** - no "new", "recently added", "now available"
- **Reference PR numbers or versions** - just state current functionality

### Adding New Content

**When adding new documentation sections:**

✅ **DO:**
- Follow the same format as existing sections
- Include complete code examples
- Explain parameters, return values, exceptions
- Show expected output
- Link to related concepts
- Add troubleshooting tips if relevant
- Include both basic and advanced usage
- **Integrate seamlessly** - new content should blend with existing docs

❌ **DON'T:**
- Add documentation without examples
- Write in a different style than existing docs
- Create orphaned sections without context
- Duplicate content that exists elsewhere
- Use different terminology for the same concepts
- **Mark sections as "new"** - just add them naturally to the appropriate location

---

## Step 4: Documentation Updates by Change Type

### New Feature Added

**Update:**
1. **README.md:** Add to Key Features if major feature
2. **API Reference:** Full documentation for new APIs
3. **Examples:** Create 1-3 examples showing usage
4. **Getting Started:** Mention if it affects initial setup

**Example Documentation Pattern:**
```markdown
### NewFeature

Description of what this feature does and why it's useful.

**Parameters:**
- `param` (type): Description

**Returns:**
- type: Description

**Example:**
```language
# Basic usage
code_example()

# Advanced usage
code_example_with_options()
```

**See Also:**
- Related features
```

### API Modified (Breaking Change)

**Update:**
1. **README.md:** Update quick start if affected
2. **API Reference:** Update signatures, parameters, behavior with current information
3. **Examples:** Update all affected examples to use new API
4. **Getting Started:** Update if affects basic usage

> **Note:** As an experimental project, simply update documentation to reflect the current API. No need to document migration paths from old versions.

### API Modified (Non-Breaking)

**Update:**
1. **API Reference:** Add new parameters with defaults
2. **Examples:** Add examples showing new parameters
3. **Advanced Topics:** Add if introduces new patterns

### Bug Fix

**Update:**
1. **Troubleshooting:** Remove workaround if it existed
2. **Examples:** Update if example demonstrated buggy behavior
3. **API Reference:** Clarify correct behavior if needed

### Performance Improvement

**Update:**
1. **Advanced Topics:** Update performance guidance
2. **API Reference:** Update if API changed
3. **Examples:** Update if best practices changed

### Deprecation

**Update:**
1. **API Reference:** Note if feature is removed, update if behavior changed
2. **Examples:** Update to current approach

> **Note:** In experimental phase, features may be removed without deprecation warnings. Simply update docs to reflect current state.

### Dependency Changes

**Update:**
1. **README.md:** Update installation if needed
2. **Getting Started:** Update prerequisites

---

## Step 5: Quality Checks

Before submitting documentation updates:

### ✓ Code Examples
- [ ] All examples are complete and runnable
- [ ] Examples use the new/updated APIs correctly
- [ ] Expected output is shown
- [ ] Examples follow best practices

### ✓ Accuracy
- [ ] API signatures match actual code
- [ ] Parameter descriptions are correct
- [ ] Return types are accurate
- [ ] Exceptions are documented correctly

### ✓ Completeness
- [ ] All new public APIs are documented
- [ ] All modified APIs reflect changes
- [ ] Breaking changes have migration guides

### ✓ Consistency
- [ ] Same terminology used throughout
- [ ] Same code style in all examples
- [ ] Same formatting as existing docs
- [ ] Internal links still work

### ✓ Clarity
- [ ] New concepts are explained clearly
- [ ] Examples are easy to understand
- [ ] No jargon without explanation
- [ ] Step-by-step for complex topics

---

## Step 6: Document Your Changes

In your PR description or commit message, note:

**Documentation Changes Made:**
- Files updated (README.md, specific docs/ files)
- What was added/modified/removed
- Reasoning for changes
- Any open questions or areas needing review

**Example:**
```markdown
## Documentation Updates

### Modified Files:
- README.md: Updated quick start example for new API
- docs/api-reference.md: Added documentation for `NewClass`
- docs/examples.md: Added 2 examples for new feature

### Changes:
- Documented new `NewClass` with parameters, returns, and exceptions
- Added basic and advanced usage examples
- Updated quick start to show new initialization pattern
- Added migration note for changed `OldFunction` signature

### Notes:
- Old examples still work but added new recommended patterns
- No breaking changes in this PR
```

---

## Common Scenarios

### Scenario: Small Bug Fix
**Action:** Update troubleshooting if there was a known workaround

### Scenario: New Public Class/Function
**Action:** Full API reference entry + 2-3 examples + possibly README mention

### Scenario: Changed Function Signature
**Action:** Update API reference + update all examples + README if in quick start

### Scenario: New Configuration Option
**Action:** Update getting started + API reference + example showing usage

### Scenario: Deprecated Feature
**Action:** Update API reference to remove old feature + update examples

### Scenario: Performance Optimization (No API Change)
**Action:** Update advanced topics with new benchmarks/patterns if applicable

---

## Documentation Style Guide

**Writing:**
- Use present tense ("returns" not "will return")
- Use active voice ("call the function" not "the function should be called")
- Be concise but complete
- Use "you" to address the reader
- Define acronyms on first use

**Code Examples:**
- Use meaningful variable names
- Include necessary imports
- Show expected output in comments or separate block
- Comment non-obvious parts
- Keep examples focused (one concept per example)

**Formatting:**
- Use `inline code` for code references
- Use ```language blocks for code examples
- Use **bold** for emphasis sparingly
- Use > for important notes/warnings
- Use proper heading hierarchy (##, ###, ####)

---

## Project-Specific Guidelines for Acton Agent

### Code Example Format
All Python examples in this project should:
- Include necessary imports from `acton_agent`
- Show complete, runnable code
- Use realistic variable names
- Include comments explaining non-obvious behavior
- Show expected output where helpful

### API Documentation Format
When documenting new classes, functions, or methods:
- Follow the existing docstring format (Google-style)
- Include Parameters, Returns, Raises sections
- Provide at least one usage example
- Cross-reference related functionality

### Feature Documentation
When adding a new feature:
- Create an example file in `examples/` directory
- Add entry to README.md if it's a major feature
- Include both basic and advanced usage patterns
- Explain the use case and benefits

---

## Now Begin

1. **Read the PR changes** - Understand what code changed
2. **Review existing documentation** - Know the current state
3. **Identify impacts** - Which docs need updates?
4. **Make updates** - Modify relevant documentation files
5. **Verify quality** - Check examples, accuracy, completeness
6. **Document changes** - List what you updated and why

Maintain the existing documentation style and structure while ensuring all content accurately reflects the new code.

