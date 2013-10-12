import json
import urllib
import dateutil.parser
from collections import OrderedDict
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import blended_transform_factory


cache = '_pr_cache.txt'

# Obtain release dates using, e.g.,
#
#  git log v0.4 -n 1 --format='%ai'
#
# The first two releases are commented out.
# This was in the era before PRs.
#
releases = OrderedDict([
    #('0.1', u'2009-10-07 13:52:19 +0200'),
    #('0.2', u'2009-11-12 14:48:45 +0200'),
    #('0.3', u'2011-10-10 03:28:47 -0700'),
    ('0.4', u'2011-12-03 14:31:32 -0800'),
    ('0.5', u'2012-02-26 21:00:51 -0800'),
    ('0.6', u'2012-06-24 21:37:05 -0700'),
    ('0.7', u'2012-09-29 18:08:49 -0700'),
    ('0.8', u'2013-03-04 20:46:09 +0100')])


month_duration = 24


def fetch_PRs(user='scikit-image', repo='scikit-image', state='open'):
    params = {'state': state,
              'per_page': 100,
              'page': 1}

    data = []
    page_data = True

    while page_data:
        config = {'user': user,
                  'repo': repo,
                  'params': urllib.urlencode(params)}

        fetch_status = ('Fetching page %(page)d (state=%(state)s)' % params +
                        ' from %(user)s/%(repo)s...' % config)
        print(fetch_status)

        f = urllib.urlopen(
            'https://api.github.com/repos/%(user)s/%(repo)s/pulls?%(params)s'
            % config
        )

        params['page'] += 1

        page_data = json.loads(f.read())

        if 'message' in page_data and page_data['message'] == "Not Found":
            page_data = []
            print('Warning: Repo not found (%(user)s/%(repo)s)' % config)
        else:
            data.extend(page_data)

    return data


def seconds_from_epoch(dates):
    seconds = [(dt - epoch).total_seconds() for dt in dates]
    return seconds


def get_month_bins(dates):
    now = datetime.now(tz=dates[0].tzinfo)
    this_month = datetime(year=now.year, month=now.month, day=1,
                          tzinfo=dates[0].tzinfo)

    bins = [this_month - relativedelta(months=i)
            for i in reversed(range(-1, month_duration))]
    return seconds_from_epoch(bins)


def date_formatter(value, _):
    dt = epoch + timedelta(seconds=value)
    return dt.strftime('%Y/%m')


for r in releases:
    releases[r] = dateutil.parser.parse(releases[r])


try:
    PRs = json.loads(open(cache, 'r').read())
    print('Loaded PRs from cache...')

except IOError:
    PRs = fetch_PRs(user='stefanv', repo='scikits.image', state='closed')
    PRs.extend(fetch_PRs(state='open'))
    PRs.extend(fetch_PRs(state='closed'))

    cf = open(cache, 'w')
    cf.write(json.dumps(PRs))
    cf.flush()

nrs = [pr['number'] for pr in PRs]
print('Processing %d pull requests...' % len(nrs))

dates = [dateutil.parser.parse(pr['created_at']) for pr in PRs]

epoch = datetime(2009, 1, 1, tzinfo=dates[0].tzinfo)

dates_f = seconds_from_epoch(dates)
bins = get_month_bins(dates)

fig, ax = plt.subplots(figsize=(7, 5))

n, bins, _ = ax.hist(dates_f, bins=bins, color='blue', alpha=0.6)

ax.xaxis.set_major_formatter(FuncFormatter(date_formatter))
ax.set_xticks(bins[2:-1:3])  # Date label every 3 months.

labels = ax.get_xticklabels()
for l in labels:
    l.set_rotation(40)
    l.set_size(10)

mixed_transform = blended_transform_factory(ax.transData, ax.transAxes)

for version, date in releases.items():
    date = seconds_from_epoch([date])[0]
    ax.axvline(date, color='black', linestyle=':', label=version)
    ax.text(date, 1, version, color='r', va='bottom', ha='center',
            transform=mixed_transform)

ax.set_title('Pull request activity').set_y(1.05)
ax.set_xlabel('Date')
ax.set_ylabel('PRs per month', color='blue')
fig.subplots_adjust(top=0.875, bottom=0.225)

cumulative = np.cumsum(n)
cumulative += len(dates) - cumulative[-1]

ax2 = ax.twinx()
ax2.plot(bins[1:], cumulative, color='black', linewidth=2)
ax2.set_ylabel('Total PRs', color='black')

fig.savefig('PRs.png')

plt.show()
