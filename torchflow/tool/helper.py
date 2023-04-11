import datetime


def datetime_str():
    return '{date:%Y-%m-%d-%H-%M-%S}'.format(date=datetime.datetime.now())


def module_str(module):
    row_format = '  {name:<40} {shape:>20} = {total_size:>12,d}'
    lines = ['  ' + '-' * 84,]

    params = list(module.named_parameters())
    for name, param in params:
        lines.append(row_format.format(name=name,
            shape=' * '.join(str(p) for p in param.size()), total_size=param.numel()))

    lines.append('  ' + '-' * 84)
    lines.append(row_format.format(name='all parameters', shape='sum of above',
        total_size=sum(int(param.numel()) for name, param in params)))
    lines.append('  ' + '=' * 84)
    lines.append('')

    return '\n'.join(lines)