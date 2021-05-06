$(document).ready(function () {
    $('#main-content-button-1').click(function () {
        var url = '/fragment/index_content';

        $('#replace_section').load(url, function( response, status, xhr ) {
            if (status === "error") {
                location.href = '/error/404';
            }
        });
    });

    $('#main-content-button-2').click(function () {
        var url = '/fragment/index_content';

        $('#replace_section').load(url, function( response, status, xhr ) {
            if (status === "error") {
                location.href = '/error/404';
            }
        });
    });

    $('#profile-content-button').click(function () {
        var url = '/fragment/profile_content';

        $('#replace_section').load(url, function( response, status, xhr ) {
            if (status === "error") {
                location.href = '/error/404';
            }
        });
    });

    $('#settings-content-button').click(function () {
        var url = '/fragment/settings_content';

        $('#replace_section').load(url, function( response, status, xhr ) {
            if (status === "error") {
                location.href = '/error/404';
            }
        });
    });

    $('#activity-content-button').click(function () {
        var url = '/fragment/activity_content';

        $('#replace_section').load(url, function( response, status, xhr ) {
            if (status === "error") {
                location.href = '/error/404';
            }
        });
    });

    $('#capsnet-content-button').click(function () {
        var url = '/ml/capsnet_content';

        $('#replace_section').load(url, function( response, status, xhr ) {
            if (status === "error") {
                location.href = '/error/404';
            }
        });
    });

    $('#video-content-button').click(function () {
        var url = '/fragment/video_content';

        $('#replace_section').load(url, function( response, status, xhr ) {
            if (status === "error") {
                location.href = '/error/404';
            }
        });
    });

    $('#learning-stat-content-button').click(function () {
        var url = '/fragment/examples/animation_content';

        $('#replace_section').load(url, function( response, status, xhr ) {
            if (status === "error") {
                location.href = '/error/404';
            }
        });
    });

    $('#detection-stat-content-button').click(function () {
        var url = '/fragment/examples/border_content';

        $('#replace_section').load(url, function( response, status, xhr ) {
            if (status === "error") {
                location.href = '/error/404';
            }
        });
    });

    $('#classification-stat-content-button').click(function () {
        var url = '/fragment/examples/color_content';

        $('#replace_section').load(url, function( response, status, xhr ) {
            if (status === "error") {
                location.href = '/error/404';
            }
        });
    });

    $('#common-stat-content-button').click(function () {
        var url = '/fragment/examples/other_content';

        $('#replace_section').load(url, function( response, status, xhr ) {
            if (status === "error") {
                location.href = '/error/404';
            }
        });
    });

    $('#charts-content-button').click(function () {
        var url = '/fragment/examples/charts_content';

        $('#replace_section').load(url, function( response, status, xhr ) {
            if (status === "error") {
                location.href = '/error/404';
            }
        });
    });
})