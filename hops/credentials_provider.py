"""
AWS temporary credential provider.

"""

from hops import constants, util, hdfs
from hops.exceptions import RestAPIError
import uuid


def assume_role(role, role_session_name=" ", duration_seconds=3600):
    """
    Assume a role and sets the temporary credential to the spark context hadoop configuration.

    Args:
        :role: (string) the role arn to be assumed
        :role_session_name: (string) use to uniquely identify a session when the same role is assumed by different principals or for different reasons.
        :duration_seconds: (int) the duration of the session. Maximum session duration is 3600 seconds.

    >>> from hops.credentials_provider import assume_role
    >>> assume_role("arn:aws:iam::<AccountNumber>:role/analyst")

    Returns:
        temporary credentials
    """
    query = "?" + constants.REST_CONFIG.HOPSWORKS_CLOUD_SESSION_TOKEN_RESOURCE_QUERY_ROLE + "=" + role
    query += "&" + constants.REST_CONFIG.HOPSWORKS_CLOUD_SESSION_TOKEN_RESOURCE_QUERY_SESSION + "="
    query += role_session_name if role_session_name != " " else "session_" + uuid.uuid4().__str__()
    if duration_seconds != 3600:
        query += "&" + constants.REST_CONFIG.HOPSWORKS_CLOUD_SESSION_TOKEN_RESOURCE_QUERY_SESSION_DURATION + "=" \
                + duration_seconds
    method = constants.HTTP_CONFIG.HTTP_GET
    resource_url = constants.DELIMITERS.SLASH_DELIMITER + constants.REST_CONFIG.HOPSWORKS_REST_RESOURCE + \
        constants.DELIMITERS.SLASH_DELIMITER + constants.REST_CONFIG.HOPSWORKS_CLOUD_RESOURCE + \
        constants.DELIMITERS.SLASH_DELIMITER + \
        constants.REST_CONFIG.HOPSWORKS_CLOUD_SESSION_TOKEN_RESOURCE + query

    response = util.send_request(method, resource_url)
    json_content = _parse_response(response, resource_url)
    sc = util._find_spark().sparkContext
    sc._jsc.hadoopConfiguration().set(constants.S3_CONFIG.S3_CREDENTIAL_PROVIDER_ENV,
                                      constants.S3_CONFIG.S3_TEMPORARY_CREDENTIAL_PROVIDER)
    sc._jsc.hadoopConfiguration().set(constants.S3_CONFIG.S3_ACCESS_KEY_ENV,
                                      json_content[constants.REST_CONFIG.JSON_ACCESS_KEY_ID])
    sc._jsc.hadoopConfiguration().set(constants.S3_CONFIG.S3_SECRET_KEY_ENV,
                                      json_content[constants.REST_CONFIG.JSON_SECRET_KEY_ID])
    sc._jsc.hadoopConfiguration().set(constants.S3_CONFIG.S3_SESSION_KEY_ENV,
                                      json_content[constants.REST_CONFIG.JSON_SESSION_TOKEN_ID])
    return json_content


def get_roles():
    """
    Get all roles mapped to the current project

    >>> from hops.credentials_provider import get_roles
    >>> get_roles()

    Returns:
        A list of role arn
    """
    json_content = _get_roles()
    items = json_content[constants.REST_CONFIG.JSON_ARRAY_ITEMS]
    roles = []
    for role in items:
        roles.append(role[constants.REST_CONFIG.JSON_CLOUD_ROLE])
    return roles


def get_role(role_id):
    """
    Get a role arn mapped to the current project by id

    Args:
        :role_id: (int) id of the role

    >>> from hops.credentials_provider import get_role
    >>> get_role(id)

    Returns:
        A role arn
    """
    json_content = _get_roles(role_id=role_id)
    return json_content[constants.REST_CONFIG.JSON_CLOUD_ROLE]


def _get_roles(role_id=None):
    by_id = ""
    if role_id:
        by_id = constants.DELIMITERS.SLASH_DELIMITER + str(role_id)
    method = constants.HTTP_CONFIG.HTTP_GET
    resource_url = constants.DELIMITERS.SLASH_DELIMITER + constants.REST_CONFIG.HOPSWORKS_REST_RESOURCE + \
        constants.DELIMITERS.SLASH_DELIMITER + constants.REST_CONFIG.HOPSWORKS_PROJECT_RESOURCE + \
        constants.DELIMITERS.SLASH_DELIMITER + hdfs.project_id() + constants.DELIMITERS.SLASH_DELIMITER + \
        constants.REST_CONFIG.HOPSWORKS_CLOUD_RESOURCE + constants.DELIMITERS.SLASH_DELIMITER + \
        constants.REST_CONFIG.HOPSWORKS_CLOUD_ROLE_MAPPINGS_RESOURCE + by_id
    response = util.send_request(method, resource_url)
    return _parse_response(response, resource_url)


def _parse_response(response, url):
    if response.ok:
        return response.json()
    else:
        error_code, error_msg, user_msg = util._parse_rest_error(response.json())
        raise RestAPIError("Error calling {}. Got status: HTTP code: {}, HTTP reason: {}, error code: {}, error msg: {}"
                           ", user msg: {}".format(url, response.status_code, response.reason, error_code, error_msg,
                                                   user_msg))
