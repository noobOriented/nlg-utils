# nlg-utils

## TqdmRedirector

### usage

```python
bar = tqdm(..., file=TqdmRedirector.STDOUT)
TqdmRedirector.enable()

print('abc')  # won't overlap bar
sys.stdout.write('def')  # won't overlap bar

TqdmRedirector.disable()

print('abc')  # will overlap bar
sys.stdout.write('def')  # will overlap bar
```
