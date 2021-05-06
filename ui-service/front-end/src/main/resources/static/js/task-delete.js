var deleteBtn = null;

function openDeleteModal(elem) {
    deleteBtn = elem;
    $('#delete-modal').modal('show');
}

$(document).ready(function() {
    $('#btn-delete').click(function () {
        var idUser = deleteBtn.getAttribute("data-iduser");
        var idTask = deleteBtn.getAttribute("data-idtask");
        var overlay = $('#overlay');
        overlay.css('display', 'block');
        $('#delete-modal').modal('hide');
        $.ajax({
            url: '/auth/task/' + idTask + '/delete',
            type: "POST",
            dataType: "html",
            success: function (data) {
                $('#tasks-div').html($(data).find('#tasks-div').html());
                overlay.css('display', 'none');
            }, error: function () {
                overlay.css('display', 'none');
                $('#info-text-modal').text("При удалении произошла ошибка. Повторите позже.");
                $('#btn-ok').click(function () {
                    $('#info-modal').modal('hide');
                });
                $('#info-modal').modal('show');
            }
        })
    });
});