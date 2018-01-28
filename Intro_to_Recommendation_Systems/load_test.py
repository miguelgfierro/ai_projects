import aiohttp
import asyncio
import json


def chunked_http_client(num_chunks, s):
    # Use semaphore to limit number of requests
    semaphore = asyncio.Semaphore(num_chunks)
    @asyncio.coroutine
    # Return co-routine that will work asynchronously and respect locking of semaphore
    def http_get(url, payload, verbose):
        nonlocal semaphore
        with (yield from semaphore):
            headers = {'content-type': 'application/json'}
            response = yield from s.request('post', url, data=json.dumps(payload), headers=headers)
            if verbose: print("Response status:", response.status)
            body = yield from response.json()
            if verbose: print(body)
            yield from response.wait_for_close()
        return body
    return http_get


def run_load_test(url, payloads, _session, concurrent, verbose):
    http_client = chunked_http_client(num_chunks=concurrent, s=_session)
    
    # http_client returns futures, save all the futures to a list
    tasks = [http_client(url, payload, verbose) for payload in payloads]

    dfs_route = []
    # wait for futures to be ready then iterate over them
    for future in asyncio.as_completed(tasks):
        data = yield from future
        try:
            dfs_route.append(data)
        except Exception as err:
            print("Error {0}".format(err))
    return dfs_route