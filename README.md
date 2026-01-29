# LST CMSSW Fork

[![Nightly Testing](https://github.com/SegmentLinking/cmssw/actions/workflows/nightly-testing.yml/badge.svg)](https://github.com/SegmentLinking/cmssw/actions/workflows/nightly-testing.yml)

This fork is used for internal development and validation of the Line Segment Tracking (LST) algorithm. This particular branch is only used to configure our custom CI to check PRs. To look at the code, please view [our master branch](https://github.com/SegmentLinking/cmssw/tree/master), which tracks the [upstream master branch](https://github.com/cms-sw/cmssw) of the official CMSSW repository.

## CI Instructions

The CI is triggered by the presence of the string `run-ci: ` in PR comments. The comment must exclusively contain CI instructions and it accepts the following items.

- `run-ci`: (string or list) This is what indicates which workflows you want to run. The available options are:
  - `all`: Run all available workflows.
  - `standalone`: Run the standalone tests.
  - `cmssw`: Run the CMSSW 29834.1 workflow.
  - `checks`: Run the SCRAM checks.

- `required-prs`: (int or list) PR number or list of PR numbers indicating required PRs that must be merged before checks and tests are run.

- `modifiers`: (string or list) Modifier or list of modifiers that can change some parameters of how the tests run. The available options are:
  - `gpu`: Run the tests on the self-hosted GPU CI.
  - `lowpt`: Use the low pT setup.

### Examples

The most basic example is just running all the workflows with no modifications, which can be done with.

```yaml
run-ci: all
```

The following exemplifies all the available options.

```yaml
run-ci: [standalone, cmssw, checks]
required-prs: [223, 224]
modifiers: [gpu, lowpt]
```
or equivalently
```yaml
run-ci:
  - standalone
  - cmssw
  - checks
required-prs: 
  - 223
  - 224
modifiers: 
  - gpu
  - lowpt
```