# Copyright (c) YugabyteDB, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied.  See the License for the specific language governing permissions and limitations
# under the License.

"""
Unit tests for csi_report: the no-op behavior when CSI is not configured, and the query retries.

Holding a launch id is not evidence that there is a server to talk to: YB_CSI_LID can be set while
CSI_SERVER/CSI_TOKEN are not. Every entry point must then no-op rather than build
'https:///api/v2/' and raise InvalidURL('No host supplied') - which is how an unconfigured CSI once
aborted a whole test run.
"""

from typing import Any, List

import pytest

# Import the module, not TestDescriptor, so pytest does not collect the Test*-named class.
from yugabyte import csi_report
from yugabyte import test_descriptor


@pytest.fixture(autouse=True)
def no_csi(monkeypatch: pytest.MonkeyPatch) -> None:
    """CSI unset but a launch present - the combination the guards exist for."""
    monkeypatch.delenv('CSI_SERVER', raising=False)
    monkeypatch.delenv('CSI_TOKEN', raising=False)
    monkeypatch.delenv('CSI_PROJ', raising=False)
    monkeypatch.setenv('YB_CSI_LID', 'some-launch-uuid')
    monkeypatch.setenv('YB_CSI_C++', 'some-suite-uuid')

    def no_requests(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("CSI is not configured; no HTTP request should have been attempted")

    for method in ('get', 'post', 'put'):
        monkeypatch.setattr(csi_report.requests, method, no_requests)


def test_configured_needs_both_server_and_token(monkeypatch: pytest.MonkeyPatch) -> None:
    assert not csi_report.configured()
    monkeypatch.setenv('CSI_SERVER', 'csi.example.com')
    assert not csi_report.configured()          # token still missing
    monkeypatch.setenv('CSI_TOKEN', 'a-token')
    assert csi_report.configured()


def test_launch_qid_is_a_no_op() -> None:
    assert csi_report.launch_qid() == ''


def test_create_suite_returns_the_var_name_with_no_value() -> None:
    assert csi_report.create_suite(
        qid='', suite_name='C++', parent='', method='Requested', planned=1, reps=1,
        time_sec=0.0) == ('YB_CSI_C++', '')


def test_close_item_is_a_no_op() -> None:
    assert csi_report.close_item('some-suite-uuid', 0.0, '', []) == ''


def test_create_test_is_a_no_op() -> None:
    descriptor = test_descriptor.TestDescriptor('tests-x/a-test:::A.B')
    assert csi_report.create_test(descriptor, 0.0, 1, rerun=False) == ''


def test_upload_log_is_a_no_op() -> None:
    assert csi_report.upload_log('some-suite-uuid', 0.0, ['/nonexistent/log']) == 0


def test_previous_launch_unique_ids_is_a_no_op() -> None:
    # None, not an empty set: "no tests are known" would make every test read as new.
    assert csi_report.previous_launch_unique_ids() is None


# ---------------------------------------------------------------------------------------------
# get_with_retries: a transient CSI failure must not decide a query's answer.
#
# These bypass the no_csi fixture's requests.get trap by replacing it themselves - the retry
# helper is below the configured() guards and is tested directly.
# ---------------------------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.text = f"body for {status_code}"


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """The retry delay is real seconds; the test should not pay them."""
    monkeypatch.setattr(csi_report.time, 'sleep', lambda seconds: None)


def fake_get(monkeypatch: pytest.MonkeyPatch, outcomes: List[Any]) -> List[int]:
    """
    Serve one outcome per call - a FakeResponse to return or an exception to raise - and record
    how many calls were made.
    """
    calls = []

    def get(url: str, headers: Any = None, params: Any = None) -> Any:
        calls.append(1)
        outcome = outcomes[len(calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(csi_report.requests, 'get', get)
    return calls


def test_first_attempt_succeeds_without_retrying(
        monkeypatch: pytest.MonkeyPatch, no_sleep: None) -> None:
    ok = FakeResponse(200)
    calls = fake_get(monkeypatch, [ok])

    assert csi_report.get_with_retries('http://csi/x', {}, {}) is ok
    assert len(calls) == 1


@pytest.mark.parametrize('transient', [
    FakeResponse(504),                              # the gateway timeouts CSI actually returns
    csi_report.requests.RequestException('connection reset'),
])
def test_transient_failure_is_retried_then_succeeds(
        monkeypatch: pytest.MonkeyPatch, no_sleep: None, transient: Any) -> None:
    """
    A 5xx and a dropped connection are the same thing to a query: try again. Returning None here
    instead would report 'nothing measured yet' and cost a redundant re-measurement.
    """
    ok = FakeResponse(200)
    calls = fake_get(monkeypatch, [transient, ok])

    assert csi_report.get_with_retries('http://csi/x', {}, {}) is ok
    assert len(calls) == 2


def test_returns_none_once_the_attempts_are_spent(
        monkeypatch: pytest.MonkeyPatch, no_sleep: None) -> None:
    """Callers must get None - not an exception - so a CSI outage degrades instead of failing."""
    calls = fake_get(monkeypatch, [FakeResponse(500)] * csi_report.GET_ATTEMPTS)

    assert csi_report.get_with_retries('http://csi/x', {}, {}) is None
    assert len(calls) == csi_report.GET_ATTEMPTS


def test_delay_grows_with_the_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    """A busy server gets a longer pause each time round, and none after the last attempt."""
    slept: List[float] = []
    monkeypatch.setattr(csi_report.time, 'sleep', lambda seconds: slept.append(seconds))
    fake_get(monkeypatch, [FakeResponse(503)] * csi_report.GET_ATTEMPTS)

    assert csi_report.get_with_retries('http://csi/x', {}, {}) is None
    assert slept == [csi_report.GET_RETRY_DELAY_SEC * attempt
                     for attempt in range(1, csi_report.GET_ATTEMPTS)]


# ---------------------------------------------------------------------------------------------
# previous_launch_unique_ids: the launch this build's tests are "new" against.
#
# Picking the wrong launch does not produce a slightly wrong answer, it produces a flood: every
# test the picked launch did not report reads as new. So the guards are pinned here.
# ---------------------------------------------------------------------------------------------

class FakeJson(FakeResponse):
    def __init__(self, body: Any) -> None:
        super().__init__(200)
        self.body = body

    def json(self) -> Any:
        return self.body


def attrs(**pairs: str) -> List[Any]:
    return [{'key': key, 'value': value} for key, value in pairs.items()]


def items_page(unique_ids: List[str], total_elements: int, total_pages: int) -> FakeJson:
    return FakeJson({
        'content': [{'uniqueId': unique_id} for unique_id in unique_ids],
        'page': {'totalElements': total_elements, 'totalPages': total_pages},
    })


def serve_csi(monkeypatch: pytest.MonkeyPatch, current: Any, launches: Any,
              item_pages: Any = None) -> List[Any]:
    """
    A configured CSI answering the three queries the function makes: this launch, the lane's
    launches, and one page of items at a time. Records the params of every call.
    """
    monkeypatch.setenv('CSI_SERVER', 'csi.example.com')
    monkeypatch.setenv('CSI_TOKEN', 'a-token')
    monkeypatch.setenv('CSI_PROJ', 'DBFT')
    calls: List[Any] = []

    def get(url: str, headers: Any = None, params: Any = None) -> Any:
        calls.append((url, params))
        if url.endswith('/launch/some-launch-uuid'):
            return FakeJson(current)
        if url.endswith('/launch'):
            return FakeJson({'content': launches})
        if url.endswith('/item'):
            return (item_pages or [])[int(params['page.page']) - 1]
        raise AssertionError("unexpected CSI query: " + url)

    monkeypatch.setattr(csi_report.requests, 'get', get)
    return calls


def test_previous_launch_is_the_newest_completed_older_one(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Newer launches and the current one are not a baseline, and neither is one that did not run to
    the end: it reported a fraction of its tests and the rest would read as new.
    """
    current = {'id': 500, 'name': 'master-alma8-clang21-tsan',
               'attributes': attrs(version='2.31.0.0', bld='4242')}
    launches = [
        {'id': 501, 'status': 'IN_PROGRESS'},   # the next lane run, already started
        {'id': 500, 'status': 'FAILED'},        # this launch
        {'id': 499, 'status': 'INTERRUPTED'},   # aborted: an incomplete test list
        {'id': 498, 'status': 'FAILED'},        # the one to compare against
        {'id': 497, 'status': 'PASSED'},
    ]
    calls = serve_csi(monkeypatch, current, launches, [
        items_page(['a-test:::A.One', 'a-test:::A.Two'], total_elements=3, total_pages=2),
        items_page(['B#three'], total_elements=3, total_pages=2),
    ])

    assert csi_report.previous_launch_unique_ids() == {
        'a-test:::A.One', 'a-test:::A.Two', 'B#three'}

    lane_query = calls[1][1]
    assert lane_query['filter.eq.name'] == 'master-alma8-clang21-tsan'
    # Same lane name, other branch: the family name folds several branches together and only the
    # version tells them apart.
    assert lane_query['filter.has.compositeAttribute'] == 'version:2.31.0.0'
    assert calls[2][1]['filter.eq.launchId'] == '498'


def test_a_per_change_launch_has_no_lane_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    """A diff's predecessor is another revision of the same diff, not the lane's previous state."""
    current = {'id': 500, 'name': 'D12345-alma9-clang21-asan',
               'attributes': attrs(diff='3', branch='master')}
    calls = serve_csi(monkeypatch, current, [])

    assert csi_report.previous_launch_unique_ids() is None
    assert len(calls) == 1


def test_no_older_launch_means_no_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first launch of a lane: every test is new, and repeating them all is not the point."""
    current = {'id': 500, 'name': 'master-alma8-clang21-tsan',
               'attributes': attrs(version='2.31.0.0', bld='4242')}
    serve_csi(monkeypatch, current, [{'id': 500, 'status': 'FAILED'}])

    assert csi_report.previous_launch_unique_ids() is None


def test_an_oversized_item_query_is_abandoned(monkeypatch: pytest.MonkeyPatch) -> None:
    """Far more items than a lane reports is not one lane's list; paging on would be wasted."""
    current = {'id': 500, 'name': 'master-alma8-clang21-tsan',
               'attributes': attrs(version='2.31.0.0', bld='4242')}
    serve_csi(monkeypatch, current, [{'id': 498, 'status': 'PASSED'}], [
        items_page(['a-test:::A.One'],
                   total_elements=csi_report.ITEM_QUERY_LIMIT + 1, total_pages=500),
    ])

    assert csi_report.previous_launch_unique_ids() is None


# ---------------------------------------------------------------------------------------------
# classify_execution: the retry_kind decision table. Only fail_repetition vs repetition is
# exclusive by construction (--fail_repetitions is rejected alongside --num_repetitions > 1);
# a Spark task resubmit (attempt > 0) can occur inside either job, and the branch order in
# classify_execution resolves those overlaps: fail_repetition wins (the "first attempt failed"
# implication must stay exact for consumers), then task_resubmit (needs the resubmit wait),
# then repetition. The kind attribute is what lets downstream consumers separate a
# fail-repetition (which must never enter a first-attempt failure rate) from a Spark task
# resubmit (infra artifact) and a plain repetition.

@pytest.mark.parametrize('rerun,attempt,reps,attempt_index,expected', [
    # expected = (retry, retry_kind, wait)
    (False, 0, '1', 1, (False, '', False)),                # plain first attempt
    (True, 0, '1', 1, (True, 'fail_repetition', False)),   # --fail_repetitions re-run
    (True, 0, '1', 3, (True, 'fail_repetition', False)),
    (False, 1, '1', 1, (True, 'task_resubmit', True)),     # Spark re-ran a dead task
    (False, 2, '1', 1, (True, 'task_resubmit', True)),
    (False, 0, '10', 1, (True, 'repetition', False)),      # first repetition may skip the wait
    (False, 0, '10', 3, (True, 'repetition', True)),
    # Overlaps, resolved by precedence:
    (True, 1, '1', 2, (True, 'fail_repetition', False)),   # resubmit inside the rerun job
    (False, 1, '10', 3, (True, 'task_resubmit', True)),    # resubmit inside a repetitions job
])
def test_classify_execution(rerun: bool, attempt: int, reps: str, attempt_index: int,
                            expected: Any) -> None:
    assert csi_report.classify_execution(rerun, attempt, reps, attempt_index) == expected


# The new-test repetitions are dispatched serially after the main pass, and only for a test whose
# first attempt passed, so consumers may read the kind on an item as "the first attempt passed".
# That has to hold for every execution of the job, a Spark resubmit inside it included, which is
# why the kind outranks task_resubmit.
@pytest.mark.parametrize('rerun,attempt,attempt_index,expected', [
    (False, 0, 2, (True, 'new_test_repetition', False)),
    (False, 3, 5, (True, 'new_test_repetition', False)),  # resubmit inside the new-test job
    # fail_repetition still wins, so a first-attempt failure can never read as a passing birth.
    (True, 0, 2, (True, 'fail_repetition', False)),
])
def test_classify_execution_new_test(rerun: bool, attempt: int, attempt_index: int,
                                     expected: Any) -> None:
    assert csi_report.classify_execution(
        rerun, attempt, '1', attempt_index, new_test=True) == expected
