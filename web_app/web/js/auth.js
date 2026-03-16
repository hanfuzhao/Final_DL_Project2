/**
 * Authentication — simple hardcoded username/password gate.
 */
var SafeTypeAuth = {
    init: function () {},

    signIn: function (username, password) {
        return new Promise(function (resolve, reject) {
            if (username === 'username' && password === 'password') {
                sessionStorage.setItem('safetype_auth', '1');
                resolve(true);
            } else {
                reject(new Error('Invalid username or password'));
            }
        });
    },

    signOut: function () {
        sessionStorage.removeItem('safetype_auth');
    },

    getSession: function () {
        return sessionStorage.getItem('safetype_auth') === '1' ? true : null;
    }
};
