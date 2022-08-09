## Proposed changes

Describe the big picture of your changes here to communicate what they accomplish and why this pull request should be accepted.

If it fixes a bug or resolves a feature request, be sure to link to that issue with `closes #<issue number>`.

If it depends on other pull requests, be sure to list and link those dependencies.

## Types of changes

What types of changes does your code introduce?

_Replace the space with an `x` for ALL the boxes that apply_

- [ ] Bugfix (Change which fixes an issue)
- [ ] New feature (Change which adds functionality)
- [ ] Enhancement to existing functionality (non-breaking change which improves existing code or documentation)
- [ ] Breaking change (fix or feature that changes results and requires updates to test baselines)
- [ ] Documentation Update

## Checklist

_Replace the space with an `x` for ALL the boxes that apply._

_You can also fill these out after creating the PR. If you're unsure about any of them, don't hesitate to ask. We're here to help! This is simply a reminder of what we are going to look for before merging your code._

- [ ] I have read the [CONTRIBUTING](https://github.com/NOAA-GSL/SENA-shallow-water/blob/develop/CONTRIBUTING.md) document
- [ ] I have ensured all my changes contribute to a common purpose
- [ ] I have verified my change does not add debug code
- [ ] I have verified my change does not add code that is commented out
- [ ] I have verified unit tests pass locally with my changes
- [ ] I have verified my changes adhere to style guidelines
- [ ] I have added tests that prove my fix is effective or that my feature works (if applicable)
- [ ] I have added necessary documentation (if appropriate)

## Further comments

If this is a relatively large or complex change, kick off the discussion by explaining why you chose the solution you did and what alternatives you considered, etc...

---

## Reviewers' checklist

The following is a checklist for reviewers to consider when reviewing this pull request.

#### Implementation

- [ ] Does this code do what it is supposed to do?
- [ ] Can the code be simplified?
- [ ] Does the code add unwanted compile-time or run-time dependencies?
- [ ] Is the code at the correct abstraction level?
- [ ] Is the code modular enough?
- [ ] Does similar functionality already exist in the codebase? If so, why wasn't it used?
- [ ] Can you think of a different solution that is substantially better in terms of code maintenance, readability, or performance?

#### Logic Errors and Bugs

- [ ] Can you think of a use case in which the code does not behave properly?
- [ ] Can you think of any inputs or external events that could break the code?

#### Error handling and logging

- [ ] Is error handling done correctly?
- [ ] Should any logging or debuggging be added or removed?
- [ ] Are error messages user friendly?

#### Usability

- [ ] Is the API intuitive to use?
- [ ] Is the API well documented?

#### Testing and testability

- [ ] Do the tests pass?
- [ ] Is the code testable?
- [ ] Do the tests reasonably cover the code change?
- [ ] Are there additional edge cases or inputs that should be tested?

#### Performance

- [ ] Do you think the code change will negatively impact performance?
- [ ] Do you see any potential to improve the performance of the code?

#### Dependencies

- [ ] Does this change require updates to documenation, configuration?  If so, what that done?
- [ ] Does this change affect backward compatibility?

#### Readability

- [ ] Is the code easy to understand?
- [ ] Can the readability be improved with smaller routines?
- [ ] Can the readability be improved with different subroutine/variable names?
- [ ] Is the data flow understandable?
- [ ] Could comments in the code be improved to add clarity?
- [ ] Is there any commented out code?
- [ ] Does the code conform to the style guide (Fortran style compliance cannot be checked automatically)
